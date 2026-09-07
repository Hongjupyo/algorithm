"""Local web server and server-side missing-person risk assessment API.

The OpenAI API key is read from the process environment or the project .env file.
It is never sent to, or stored by, the browser.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import webbrowser
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


PROJECT_DIR = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding process variables."""
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        logging.warning("Could not read %s: %s", path, error)
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if (
            not key
            or not (key[0].isalpha() or key[0] == "_")
            or any(not (character.isalnum() or character == "_") for character in key)
        ):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not os.environ.get(key, "").strip():
            os.environ[key] = value


load_env_file(PROJECT_DIR / ".env")

HOST = "127.0.0.1"
PORT = int(os.environ.get("APP_PORT", "5500"))
SEOUL_TZ = ZoneInfo("Asia/Seoul")
FRONTEND_DIR = PROJECT_DIR / "frontend"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_REQUEST_BYTES = 64 * 1024
HTTP_TIMEOUT_SECONDS = 45

WEATHER_FIELDS = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
)

WMO_WEATHER_LABELS = {
    0: "맑음",
    1: "대체로 맑음",
    2: "부분적으로 흐림",
    3: "흐림",
    45: "안개",
    48: "상고대 안개",
    51: "약한 이슬비",
    53: "이슬비",
    55: "강한 이슬비",
    56: "약한 어는 이슬비",
    57: "강한 어는 이슬비",
    61: "약한 비",
    63: "비",
    65: "강한 비",
    66: "약한 어는 비",
    67: "강한 어는 비",
    71: "약한 눈",
    73: "눈",
    75: "강한 눈",
    77: "싸락눈",
    80: "약한 소나기",
    81: "소나기",
    82: "강한 소나기",
    85: "약한 눈 소나기",
    86: "강한 눈 소나기",
    95: "뇌우",
    96: "약한 우박을 동반한 뇌우",
    99: "강한 우박을 동반한 뇌우",
}

RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "risk_level": {
            "type": "string",
            "enum": ["critical", "high", "moderate", "low"],
        },
        "summary": {"type": "string", "maxLength": 240},
        "risk_factors": {
            "type": "array",
            "items": {"type": "string", "maxLength": 120},
            "maxItems": 4,
        },
        "immediate_actions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 140},
            "maxItems": 4,
        },
        "confidence_note": {"type": "string", "maxLength": 220},
    },
    "required": [
        "risk_score",
        "risk_level",
        "summary",
        "risk_factors",
        "immediate_actions",
        "confidence_note",
    ],
    "additionalProperties": False,
}


class ClientInputError(ValueError):
    pass


class UpstreamServiceError(RuntimeError):
    pass


def fetch_json(url: str, *, method: str = "GET", headers: dict[str, str] | None = None,
               payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, method=method, headers=headers or {})
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")[:2000]
        logging.error("Upstream HTTP %s from %s: %s", error.code, url, error_body)
        raise UpstreamServiceError(f"외부 서비스가 오류로 응답했습니다. (HTTP {error.code})") from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        logging.error("Upstream request failed for %s: %s", url, error)
        raise UpstreamServiceError("외부 서비스에 연결할 수 없습니다.") from error


def require_number(data: dict[str, Any], key: str, minimum: float, maximum: float) -> float:
    value = data.get(key)
    if isinstance(value, bool):
        raise ClientInputError(f"{key} 값이 올바르지 않습니다.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ClientInputError(f"{key} 값이 필요합니다.") from error
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ClientInputError(f"{key} 값이 허용 범위를 벗어났습니다.")
    return number


def optional_number(data: dict[str, Any], key: str, minimum: float, maximum: float) -> float | None:
    if data.get(key) in (None, ""):
        return None
    return require_number(data, key, minimum, maximum)


def clean_string(data: dict[str, Any], key: str, max_length: int = 100) -> str:
    value = data.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ClientInputError(f"{key} 값이 올바르지 않습니다.")
    return value.strip()[:max_length]


def parse_missing_datetime(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ClientInputError("실종 일시를 입력해 주세요.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ClientInputError("실종 일시 형식이 올바르지 않습니다.") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SEOUL_TZ)
    else:
        parsed = parsed.astimezone(SEOUL_TZ)
    now = datetime.now(SEOUL_TZ)
    if parsed > now + timedelta(minutes=5):
        raise ClientInputError("실종 일시는 현재보다 미래일 수 없습니다.")
    if parsed < datetime(1940, 1, 1, tzinfo=SEOUL_TZ):
        raise ClientInputError("기상 조회는 1940년 이후 일시만 지원합니다.")
    return parsed


def validate_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ClientInputError("요청 본문이 올바르지 않습니다.")
    return {
        "latitude": require_number(data, "latitude", -90, 90),
        "longitude": require_number(data, "longitude", -180, 180),
        "missing_at": parse_missing_datetime(data.get("missing_datetime")),
        "age": optional_number(data, "age", 0, 120),
        "sex": clean_string(data, "sex"),
        "subject_type": clean_string(data, "subject_type"),
        "medical_condition": clean_string(data, "medical_condition"),
        "place_type": clean_string(data, "place_type"),
        "reported_weather": clean_string(data, "weather"),
        "reported_temperature_c": optional_number(data, "temperature_c", -80, 60),
    }


def weather_url(latitude: float, longitude: float, missing_at: datetime) -> tuple[str, str]:
    common = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "hourly": ",".join(WEATHER_FIELDS),
        "timezone": "Asia/Seoul",
    }
    if missing_at.date() < (datetime.now(SEOUL_TZ).date() - timedelta(days=5)):
        common.update({"start_date": missing_at.date().isoformat(), "end_date": missing_at.date().isoformat()})
        return f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(common)}", "Open-Meteo 과거 재분석"
    common.update({"past_days": "7", "forecast_days": "1"})
    return f"{OPEN_METEO_FORECAST_URL}?{urlencode(common)}", "Open-Meteo 단기 관측·예보"


def value_at(hourly: dict[str, Any], field: str, index: int) -> float | int | None:
    values = hourly.get(field)
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def fetch_weather(context: dict[str, Any]) -> dict[str, Any]:
    url, source = weather_url(context["latitude"], context["longitude"], context["missing_at"])
    data = fetch_json(url, headers={"User-Agent": "MissingPersonAssessment/1.0"})
    hourly = data.get("hourly")
    times = hourly.get("time") if isinstance(hourly, dict) else None
    if not isinstance(times, list) or not times:
        raise UpstreamServiceError("해당 시각의 기상 데이터를 찾지 못했습니다.")

    target_naive = context["missing_at"].replace(tzinfo=None, minute=0, second=0, microsecond=0)
    parsed_times: list[datetime] = []
    for item in times:
        try:
            parsed_times.append(datetime.fromisoformat(item))
        except (TypeError, ValueError):
            parsed_times.append(datetime.max)
    index = min(range(len(parsed_times)), key=lambda item: abs(parsed_times[item] - target_naive))
    weather_code = value_at(hourly, "weather_code", index)
    return {
        "observed_at": times[index],
        "source": source,
        "condition": WMO_WEATHER_LABELS.get(int(weather_code), "알 수 없음") if weather_code is not None else "알 수 없음",
        "weather_code": weather_code,
        "temperature_c": value_at(hourly, "temperature_2m", index),
        "apparent_temperature_c": value_at(hourly, "apparent_temperature", index),
        "relative_humidity_percent": value_at(hourly, "relative_humidity_2m", index),
        "precipitation_mm": value_at(hourly, "precipitation", index),
        "wind_speed_kmh": value_at(hourly, "wind_speed_10m", index),
        "wind_gusts_kmh": value_at(hourly, "wind_gusts_10m", index),
    }


def fallback_weather(context: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "observed_at": context["missing_at"].strftime("%Y-%m-%dT%H:%M"),
        "source": "사용자 입력값 (기상 API 조회 실패)",
        "condition": context["reported_weather"] or "알 수 없음",
        "weather_code": None,
        "temperature_c": context["reported_temperature_c"],
        "apparent_temperature_c": None,
        "relative_humidity_percent": None,
        "precipitation_mm": None,
        "wind_speed_kmh": None,
        "wind_gusts_kmh": None,
        "fallback_reason": reason,
    }


def extract_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for output_item in response.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content in output_item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "refusal":
                raise UpstreamServiceError("안전 정책에 따라 분석을 생성하지 못했습니다.")
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise UpstreamServiceError("OpenAI 응답에서 분석 결과를 찾지 못했습니다.")


def normalize_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
    score = max(0, min(100, int(assessment["risk_score"])))
    assessment["risk_score"] = score
    assessment["risk_level"] = "critical" if score >= 75 else "high" if score >= 50 else "moderate" if score >= 25 else "low"
    return assessment


def request_openai_risk_assessment(context: dict[str, Any], weather: dict[str, Any], api_key: str) -> dict[str, Any]:
    now = datetime.now(SEOUL_TZ)
    elapsed_hours = max(0, (now - context["missing_at"]).total_seconds() / 3600)
    model_input = {
        "elapsed_hours": round(elapsed_hours, 1),
        "age": context["age"],
        "sex": context["sex"] or "미확인",
        "subject_type": context["subject_type"] or "미확인",
        "medical_condition": context["medical_condition"] or "미확인",
        "place_type": context["place_type"] or "미확인",
        "weather_at_missing_time": weather,
    }
    instructions = (
        "당신은 수색 의사결정을 대신하지 않는 실종자 환경 위험 참고 분석기입니다. "
        "경과 시간, 연령·질환, 장소 유형, 당시 기상과 체감온도를 바탕으로 현재의 환경 위험도를 한국어로 분석하세요. "
        "risk_score는 0점(상대적으로 낮음)부터 100점(매우 높음)까지이며 점수가 높을수록 생명·안전 위험이 크다는 뜻입니다. "
        "risk_level은 0~24 low, 25~49 moderate, 50~74 high, 75~100 critical 기준을 정확히 따르세요. "
        "생존 확률이나 사망 확률을 계산하거나 표현하지 마세요. 제공되지 않은 사실을 가정하지 말고 정보 부족과 불확실성을 명시하세요. "
        "어떤 점수도 신고·수색의 긴급도를 낮추거나 수색 중단 근거가 될 수 없으며, 즉시 112 신고와 전문 구조기관 지시 준수를 항상 행동 항목에 포함하세요. "
        "이 점수가 임상적·통계적으로 검증된 지표가 아니라는 점을 confidence_note에 분명히 밝히세요."
    )
    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"),
        "reasoning": {"effort": "low"},
        "store": False,
        "safety_identifier": "missing-person-local-risk-assessment",
        "instructions": instructions,
        "input": json.dumps(model_input, ensure_ascii=False),
        "max_output_tokens": 1200,
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "risk_assessment",
                "strict": True,
                "schema": RISK_SCHEMA,
            },
        },
    }
    response = fetch_json(
        OPENAI_RESPONSES_URL,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
    )
    try:
        assessment = json.loads(extract_output_text(response))
    except (json.JSONDecodeError, TypeError) as error:
        raise UpstreamServiceError("OpenAI 분석 결과 형식이 올바르지 않습니다.") from error
    if not isinstance(assessment, dict):
        raise UpstreamServiceError("OpenAI 분석 결과 형식이 올바르지 않습니다.")
    return normalize_assessment(assessment)


class ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class AppHandler(SimpleHTTPRequestHandler):
    server_version = "MissingPersonAssessment/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path != "/api/risk-assessment":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.handle_risk_assessment()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path == "/api/health":
            self.write_json(
                HTTPStatus.OK,
                {
                    "service": "missing-person-assessment",
                    "api_key_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
                },
            )
            return
        super().do_GET()

    def read_json_body(self) -> Any:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ClientInputError("요청 크기가 올바르지 않습니다.") from error
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ClientInputError("요청 본문이 비어 있거나 너무 큽니다.")
        try:
            return json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ClientInputError("JSON 요청 본문이 올바르지 않습니다.") from error

    def write_json(self, status: HTTPStatus, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def handle_risk_assessment(self) -> None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                self.write_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "프로젝트의 .env 파일에 OPENAI_API_KEY를 입력해 주세요."},
                )
                return
            context = validate_payload(self.read_json_body())
            try:
                weather = fetch_weather(context)
            except UpstreamServiceError as error:
                logging.warning("Weather lookup failed; using reported input: %s", error)
                weather = fallback_weather(context, str(error))
            assessment = request_openai_risk_assessment(context, weather, api_key)
            response = {
                "assessment": assessment,
                "weather": weather,
                "elapsed_hours": round(
                    max(0, (datetime.now(SEOUL_TZ) - context["missing_at"]).total_seconds() / 3600),
                    1,
                ),
                "model": os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"),
                "disclaimer": "AI 위험도 참고 점수이며 검증된 의학·통계 지표가 아닙니다. 신고와 수색은 점수와 관계없이 즉시 계속해야 합니다.",
            }
            self.write_json(HTTPStatus.OK, response)
        except ClientInputError as error:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except UpstreamServiceError as error:
            self.write_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
        except Exception:
            logging.exception("Unexpected risk assessment error")
            self.write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "분석 중 예기치 않은 오류가 발생했습니다."})

    def log_message(self, format: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), format % args)


def create_server(preferred_port: int) -> tuple[ThreadingHTTPServer, int]:
    """Bind the preferred port, falling back when an older local server owns it."""
    last_error: OSError | None = None
    for port in range(preferred_port, preferred_port + 11):
        try:
            return ExclusiveThreadingHTTPServer((HOST, port), AppHandler), port
        except OSError as error:
            error_code = getattr(error, "winerror", None) or error.errno
            # Windows may report an occupied/exclusively bound local port as
            # either WSAEACCES (10013) or WSAEADDRINUSE (10048).
            if error_code not in {98, 10013, 10048}:
                raise
            last_error = error
            logging.warning("Port %s is already in use; trying %s.", port, port + 1)
    raise SystemExit(f"Could not find a free port from {preferred_port} to {preferred_port + 10}: {last_error}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not FRONTEND_DIR.is_dir():
        raise SystemExit(f"Frontend directory not found: {FRONTEND_DIR}")
    if not os.environ.get("OPENAI_API_KEY"):
        logging.warning("OPENAI_API_KEY is not set in the environment or project .env file.")
    server, active_port = create_server(PORT)
    app_url = f"http://{HOST}:{active_port}"
    print(f"Web app: {app_url}")
    print("Press Ctrl+C to stop.")
    if os.environ.get("NO_BROWSER") != "1":
        webbrowser.open(app_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
