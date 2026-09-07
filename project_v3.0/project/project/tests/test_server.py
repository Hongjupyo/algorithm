import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "latitude": 37.5665,
            "longitude": 126.978,
            "missing_datetime": "2024-06-15T14:30",
            "age": 70,
            "sex": "남",
            "subject_type": "노인",
            "medical_condition": "치매",
            "place_type": "공원",
            "weather": "맑음",
            "temperature_c": 24,
        }

    def test_payload_validation_treats_local_time_as_seoul(self):
        context = server.validate_payload(self.payload)
        self.assertEqual(context["missing_at"].tzinfo, server.SEOUL_TZ)
        self.assertEqual(context["age"], 70)

    def test_env_file_loads_values_without_overriding_process_environment(self):
        env_path = Path("test.env")
        contents = "OPENAI_API_KEY=from-file\nOPENAI_MODEL='file-model'\n"
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "read_text", return_value=contents):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "from-process"}, clear=True):
                server.load_env_file(env_path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-process")
                self.assertEqual(os.environ["OPENAI_MODEL"], "file-model")

    def test_archive_weather_url_for_old_case(self):
        context = server.validate_payload(self.payload)
        url, source = server.weather_url(context["latitude"], context["longitude"], context["missing_at"])
        self.assertTrue(url.startswith(server.OPEN_METEO_ARCHIVE_URL))
        self.assertIn("start_date=2024-06-15", url)
        self.assertIn("Open-Meteo", source)

    def test_server_falls_back_when_preferred_port_is_busy(self):
        busy = OSError(10013, "port unavailable on Windows")
        fake_server = object()
        with patch.object(server, "ExclusiveThreadingHTTPServer", side_effect=[busy, fake_server]) as mocked_server:
            created_server, active_port = server.create_server(5500)
        self.assertIs(created_server, fake_server)
        self.assertEqual(active_port, 5501)
        self.assertEqual(mocked_server.call_args_list[1].args[0], (server.HOST, 5501))

    def test_openai_request_uses_structured_output_without_coordinates(self):
        context = server.validate_payload(self.payload)
        weather = {
            "observed_at": "2024-06-15T14:00",
            "source": "test",
            "condition": "맑음",
            "temperature_c": 24.0,
            "apparent_temperature_c": 25.0,
        }
        structured = {
            "risk_score": 72,
            "risk_level": "high",
            "summary": "참고 분석",
            "risk_factors": ["경과 시간"],
            "immediate_actions": ["즉시 112 신고"],
            "confidence_note": "검증된 확률이 아님",
        }
        fake_response = {
            "output": [{"content": [{"type": "output_text", "text": json.dumps(structured, ensure_ascii=False)}]}]
        }
        with patch.object(server, "fetch_json", return_value=fake_response) as mocked_fetch:
            assessment = server.request_openai_risk_assessment(context, weather, "test-key")
        sent_payload = mocked_fetch.call_args.kwargs["payload"]
        self.assertEqual(sent_payload["text"]["format"]["type"], "json_schema")
        self.assertEqual(sent_payload["text"]["format"]["name"], "risk_assessment")
        self.assertNotIn("latitude", sent_payload["input"])
        self.assertNotIn("longitude", sent_payload["input"])
        self.assertEqual(assessment["risk_score"], 72)

    def test_future_missing_time_is_rejected(self):
        future_payload = dict(self.payload, missing_datetime="2999-01-01T00:00")
        with self.assertRaises(server.ClientInputError):
            server.validate_payload(future_payload)


if __name__ == "__main__":
    unittest.main()
