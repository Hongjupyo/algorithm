const GEOCODING_BASE_URL = "https://nominatim.openstreetmap.org";
const GEOCODING_INTERVAL_MS = 1100;
const LOADING_DURATION_MS = 3000;

const state = { model: null, preprocessing: null, map: null, mapBounds: null, seenMarker: null, predictionMarker: null, predictionCircle: null, routeLine: null, landmarkSearchId: 0, riskRequestId: 0, selectedPlaceName: "서울특별시청", geocodingCache: new Map(), lastGeocodingRequestAt: 0 };
const statusElement = document.querySelector("#model-status");
const predictButton = document.querySelector("#predict-button");
const form = document.querySelector("#prediction-form");
const placeInput = document.querySelector("#last_seen_place");
const placeSearchButton = document.querySelector("#place-search-button");
const placeSearchStatus = document.querySelector("#place-search-status");
const placeSearchResults = document.querySelector("#place-search-results");
const screens = { search: document.querySelector("#search-screen"), loading: document.querySelector("#loading-screen"), result: document.querySelector("#result-screen") };
const fieldIds = { sex: "sex", subject_type: "subject_type", medical_condition: "medical_condition", place_type: "place_type", weather: "weather" };

function showScreen(name) {
  Object.entries(screens).forEach(([key, element]) => { element.hidden = key !== name; });
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (name === "result" && state.map) {
    setTimeout(() => {
      state.map.relayout();
      if (state.mapBounds) {
        state.map.setBounds(state.mapBounds, 70, 70, 70, 70);
        // Prevent nearly identical points from producing an excessively close view.
        if (state.map.getLevel() < 4) state.map.setLevel(4);
      }
    }, 50);
  }
}
function setStatus(type, message) { statusElement.className = "status " + type; statusElement.querySelector("span:last-child").textContent = message; }
function setPlaceStatus(type, message) { placeSearchStatus.className = "search-status " + type; placeSearchStatus.textContent = message; }
function wait(milliseconds) { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }
function setSelectValue(id, value) { const select = document.getElementById(id); if ([...select.options].some((option) => option.value === value)) select.value = value; }
function loadKakaoMaps() { return new Promise((resolve, reject) => { const appKey = window.KAKAO_MAP_APP_KEY; if (!appKey) { reject(new Error("Kakao 지도 JavaScript 키가 설정되지 않았습니다.")); return; } const script = document.createElement("script"); script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${encodeURIComponent(appKey)}&libraries=services&autoload=false`; script.onload = () => kakao.maps.load(resolve); script.onerror = () => reject(new Error("Kakao 지도 SDK를 불러오지 못했습니다.")); document.head.append(script); }); }

function fillSelects() { const config = state.preprocessing.categorical; config.columns.forEach((column, index) => { const select = document.getElementById(fieldIds[column]); select.replaceChildren(new Option("모름 / 선택 안 함", "")); config.categories[index].forEach((value) => select.append(new Option(value, value))); }); }
async function loadAssets() { try { if (typeof tf === "undefined") throw new Error("TensorFlow.js를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요."); const response = await fetch("./model/preprocessing.json"); if (!response.ok) throw new Error("전처리 설정 파일을 불러오지 못했습니다."); state.preprocessing = await response.json(); fillSelects(); setStatus("loading", "모델과 지도를 불러오는 중"); const [model] = await Promise.all([tf.loadLayersModel("./model/model.json"), loadKakaoMaps()]); state.model = model; if (state.model.inputs[0].shape.at(-1) !== state.preprocessing.modelInputSize) throw new Error("모델과 전처리 입력 크기가 일치하지 않습니다."); setStatus("ready", "모델과 지도 준비 완료"); predictButton.disabled = false; } catch (error) { console.error(error); setStatus("error", "지도 설정 필요"); alert(`${error.message}\n\njs/kakao-config.js에 JavaScript 키를 입력하고 Kakao Developers에 현재 실행 도메인을 등록해 주세요.`); } }
async function geocodingFetch(url, cacheKey) { if (state.geocodingCache.has(cacheKey)) return state.geocodingCache.get(cacheKey); const elapsed = Date.now() - state.lastGeocodingRequestAt; if (elapsed < GEOCODING_INTERVAL_MS) await wait(GEOCODING_INTERVAL_MS - elapsed); state.lastGeocodingRequestAt = Date.now(); const response = await fetch(url, { headers: { "Accept-Language": "ko" } }); if (!response.ok) throw new Error("주소 검색 서비스가 응답하지 않습니다."); const data = await response.json(); state.geocodingCache.set(cacheKey, data); return data; }
function choosePlace(result) { const latitude = Number(result.lat), longitude = Number(result.lon); if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) { setPlaceStatus("error", "선택한 장소의 좌표가 올바르지 않습니다."); return; } placeInput.value = result.display_name; document.querySelector("#last_seen_latitude").value = latitude.toFixed(6); document.querySelector("#last_seen_longitude").value = longitude.toFixed(6); state.selectedPlaceName = result.display_name; placeSearchResults.hidden = true; placeSearchResults.replaceChildren(); setPlaceStatus("success", `좌표가 설정되었습니다. ${latitude.toFixed(6)}, ${longitude.toFixed(6)}`); }
function renderPlaceResults(results) { placeSearchResults.replaceChildren(); results.forEach((result) => { const button = document.createElement("button"); button.type = "button"; button.className = "search-result-button"; button.textContent = result.display_name; button.addEventListener("click", () => choosePlace(result)); placeSearchResults.append(button); }); placeSearchResults.hidden = false; }
async function searchPlace() { const query = placeInput.value.trim(); if (query.length < 2) { setPlaceStatus("error", "장소명을 두 글자 이상 입력해 주세요."); return; } placeSearchButton.disabled = true; placeSearchButton.textContent = "검색 중..."; setPlaceStatus("", "검색 결과를 불러오는 중입니다."); placeSearchResults.hidden = true; try { const parameters = new URLSearchParams({ format: "jsonv2", q: query, countrycodes: "kr", limit: "5", addressdetails: "1", "accept-language": "ko" }); const results = await geocodingFetch(`${GEOCODING_BASE_URL}/search?${parameters}`, `search:${query}`); if (!Array.isArray(results) || !results.length) { setPlaceStatus("error", "검색 결과가 없습니다. 더 넓은 지역명으로 검색해 보세요."); return; } renderPlaceResults(results); setPlaceStatus("", "아래 결과 중 정확한 장소를 선택해 주세요."); } catch (error) { console.error(error); setPlaceStatus("error", error.message); } finally { placeSearchButton.disabled = false; placeSearchButton.textContent = "장소 검색"; } }
async function reverseGeocode(latitude, longitude) { try { const parameters = new URLSearchParams({ format: "jsonv2", lat: latitude.toFixed(6), lon: longitude.toFixed(6), zoom: "16", addressdetails: "1", "accept-language": "ko" }); const result = await geocodingFetch(`${GEOCODING_BASE_URL}/reverse?${parameters}`, `reverse:${latitude.toFixed(5)},${longitude.toFixed(5)}`); return result.display_name || "지역명을 찾지 못했습니다."; } catch (error) { console.warn(error); return "지역명 조회 실패 - 아래 좌표를 확인해 주세요."; } }
function optionalNumberValue(id) { const rawValue = document.getElementById(id).value.trim(); if (!rawValue) return null; const value = Number(rawValue); return Number.isFinite(value) ? value : null; }
function renderCoreInputSummary() { const value = (id) => document.getElementById(id)?.value.trim() || "-"; const selectText = (id) => { const select = document.getElementById(id); return select?.options[select.selectedIndex]?.textContent?.trim() || "-"; }; const set = (id, text) => { const element = document.getElementById(id); if (element) element.textContent = text || "-"; }; set("core-age", value("age")); set("core-sex", selectText("sex")); set("core-subject-type", selectText("subject_type")); set("core-medical-condition", selectText("medical_condition")); set("core-height", value("height_cm") === "-" ? "-" : `${value("height_cm")} cm`); set("core-weight", value("weight_kg") === "-" ? "-" : `${value("weight_kg")} kg`); set("core-place", value("last_seen_place")); const weather = selectText("weather"), temperature = value("temperature_c"); set("core-weather", [weather === "모름 / 선택 안 함" ? "" : weather, temperature === "-" ? "" : `${temperature}°C`].filter(Boolean).join(" · ") || "-"); }
function preprocessInput() { const rawDate = document.querySelector("#missing_datetime").value; const missingDate = new Date(rawDate); const hasDate = !Number.isNaN(missingDate.getTime()); const numericValues = [optionalNumberValue("age"), optionalNumberValue("height_cm"), optionalNumberValue("weight_kg"), optionalNumberValue("temperature_c"), optionalNumberValue("last_seen_latitude"), optionalNumberValue("last_seen_longitude"), hasDate ? missingDate.getFullYear() : null, hasDate ? missingDate.getMonth() + 1 : null, hasDate ? missingDate.getDate() : null, hasDate ? missingDate.getHours() : null, hasDate ? (missingDate.getDay() + 6) % 7 : null]; const numericFeatures = numericValues.map((value, index) => value === null ? 0 : (value - state.preprocessing.numeric.mean[index]) / state.preprocessing.numeric.scale[index]); const categoricalFeatures = []; state.preprocessing.categorical.columns.forEach((column, index) => { const selected = document.getElementById(fieldIds[column]).value; state.preprocessing.categorical.categories[index].forEach((category) => categoricalFeatures.push(category === selected ? 1 : 0)); }); const features = numericFeatures.concat(categoricalFeatures); if (features.length !== state.preprocessing.modelInputSize) throw new Error("생성된 모델 입력 크기가 올바르지 않습니다."); return features; }
function haversineKm(lat1, lon1, lat2, lon2) { const radians = (degrees) => degrees * Math.PI / 180, earthRadiusKm = 6371, latDifference = radians(lat2 - lat1), lonDifference = radians(lon2 - lon1); const a = Math.sin(latDifference / 2) ** 2 + Math.cos(radians(lat1)) * Math.cos(radians(lat2)) * Math.sin(lonDifference / 2) ** 2; return earthRadiusKm * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)); }
function updateMap(seenLatitude, seenLongitude, latitude, longitude, predictedAddress) { const container = document.querySelector("#map"); const seenPosition = new kakao.maps.LatLng(seenLatitude, seenLongitude); const predictedPosition = new kakao.maps.LatLng(latitude, longitude); if (!state.map) state.map = new kakao.maps.Map(container, { center: predictedPosition, level: 8 }); [state.seenMarker, state.predictionMarker].forEach((marker) => marker?.setMap(null)); if (state.routeLine) state.routeLine.setMap(null); state.seenMarker = new kakao.maps.Marker({ position: seenPosition, title: `마지막 목격: ${state.selectedPlaceName || "직접 입력한 좌표"}` }); state.predictionMarker = new kakao.maps.Marker({ position: predictedPosition, title: `예상 발견: ${predictedAddress}` }); state.routeLine = new kakao.maps.Polyline({ path: [seenPosition, predictedPosition], strokeWeight: 3, strokeColor: "#2f6fed", strokeOpacity: .65, strokeStyle: "shortdash" }); state.seenMarker.setMap(state.map); state.predictionMarker.setMap(state.map); state.routeLine.setMap(state.map); const bounds = new kakao.maps.LatLngBounds(); bounds.extend(seenPosition); bounds.extend(predictedPosition); state.map.setBounds(bounds, 50, 50, 50, 50); }

function riskPayload(latitude, longitude) {
  return {
    latitude,
    longitude,
    missing_datetime: document.querySelector("#missing_datetime").value,
    age: optionalNumberValue("age"),
    sex: document.querySelector("#sex").value,
    subject_type: document.querySelector("#subject_type").value,
    medical_condition: document.querySelector("#medical_condition").value,
    place_type: document.querySelector("#place_type").value,
    weather: document.querySelector("#weather").value,
    temperature_c: optionalNumberValue("temperature_c"),
  };
}

function resetRiskAssessment() {
  state.riskRequestId += 1;
  const section = document.querySelector("#risk-assessment");
  section.className = "risk-assessment is-loading";
  document.querySelector("#risk-status-badge").textContent = "분석 중";
  document.querySelector("#risk-loading").hidden = false;
  document.querySelector("#risk-content").hidden = true;
  document.querySelector("#risk-error").hidden = true;
}

function replaceList(selector, items) {
  const list = document.querySelector(selector);
  list.replaceChildren();
  (items.length ? items : ["확인된 항목이 없습니다."]).forEach((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    list.append(item);
  });
}

function compactWeatherDetails(weather, elapsedHours) {
  const details = [];
  if (Number.isFinite(weather.temperature_c)) details.push(`${Number(weather.temperature_c).toFixed(1)}°C`);
  if (Number.isFinite(weather.apparent_temperature_c)) details.push(`체감 ${Number(weather.apparent_temperature_c).toFixed(1)}°C`);
  if (Number.isFinite(weather.relative_humidity_percent)) details.push(`습도 ${Math.round(weather.relative_humidity_percent)}%`);
  if (Number.isFinite(weather.wind_speed_kmh)) details.push(`바람 ${Number(weather.wind_speed_kmh).toFixed(1)}km/h`);
  details.push(`경과 ${Number(elapsedHours).toLocaleString("ko-KR", { maximumFractionDigits: 1 })}시간`);
  return details.join(" · ");
}

function renderRiskAssessment(result) {
  const assessment = result.assessment;
  const riskLabels = { critical: "위험 매우 높음", high: "위험 높음", moderate: "주의", low: "상대적 위험 낮음", unknown: "판단 불가" };
  const section = document.querySelector("#risk-assessment");
  section.className = `risk-assessment risk-${assessment.risk_level}`;
  document.querySelector("#risk-status-badge").textContent = riskLabels[assessment.risk_level] || "참고 분석";
  document.querySelector("#risk-score").textContent = assessment.risk_score;
  document.querySelector("#weather-condition").textContent = result.weather.condition || "알 수 없음";
  document.querySelector("#weather-details").textContent = compactWeatherDetails(result.weather, result.elapsed_hours);
  document.querySelector("#weather-source").textContent = `${result.weather.source} · ${result.weather.observed_at}`;
  document.querySelector("#risk-summary").textContent = assessment.summary;
  replaceList("#risk-factor-list", assessment.risk_factors || []);
  replaceList("#immediate-action-list", assessment.immediate_actions || []);
  document.querySelector("#assessment-confidence").textContent = assessment.confidence_note;
  document.querySelector("#assessment-disclaimer").textContent = result.disclaimer;
  document.querySelector("#risk-loading").hidden = true;
  document.querySelector("#risk-error").hidden = true;
  document.querySelector("#risk-content").hidden = false;
}

function renderRiskError(message) {
  const section = document.querySelector("#risk-assessment");
  section.className = "risk-assessment has-error";
  document.querySelector("#risk-status-badge").textContent = "분석 실패";
  document.querySelector("#risk-loading").hidden = true;
  document.querySelector("#risk-content").hidden = true;
  document.querySelector("#risk-error").hidden = false;
  document.querySelector("#risk-error-message").textContent = message;
}

async function requestRiskAssessment(latitude, longitude) {
  resetRiskAssessment();
  const requestId = state.riskRequestId;
  try {
    const response = await fetch("/api/risk-assessment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(riskPayload(latitude, longitude)),
    });
    let result;
    try { result = await response.json(); } catch { result = {}; }
    if (!response.ok) {
      if (response.status === 404 || response.status === 405) {
        throw new Error("현재 페이지가 구형 정적 서버로 실행 중입니다. 기존 서버 창을 닫고 프로젝트의 run_frontend.bat를 다시 실행해 주세요.");
      }
      throw new Error(result.error || `위험도 분석 요청에 실패했습니다. (HTTP ${response.status})`);
    }
    if (requestId === state.riskRequestId) renderRiskAssessment(result);
  } catch (error) {
    console.error(error);
    if (requestId === state.riskRequestId) renderRiskError(error.message);
  }
}

async function predict(event) { event.preventDefault(); if (!state.model) return; const placeName = placeInput.value.trim(), seenLatitude = optionalNumberValue("last_seen_latitude"), seenLongitude = optionalNumberValue("last_seen_longitude"); if (placeName && (seenLatitude === null || seenLongitude === null)) { setPlaceStatus("error", "입력한 장소를 먼저 검색해 좌표를 설정해 주세요."); placeInput.focus(); return; } predictButton.disabled = true; const startedAt = Date.now(); showScreen("loading"); try { const features = preprocessInput(); const scaledOutput = tf.tidy(() => state.model.predict(tf.tensor2d([features])).dataSync()); const [latitude, longitude] = scaledOutput.map((value, index) => value * state.preprocessing.target.scale[index] + state.preprocessing.target.mean[index]); const sourceLatitude = seenLatitude ?? state.preprocessing.numeric.mean[4], sourceLongitude = seenLongitude ?? state.preprocessing.numeric.mean[5], distance = haversineKm(sourceLatitude, sourceLongitude, latitude, longitude); const latitudeText = latitude.toFixed(6), longitudeText = longitude.toFixed(6), distanceText = `${distance.toFixed(1)} km`; document.querySelector("#result-latitude").textContent = latitudeText; document.querySelector("#result-longitude").textContent = longitudeText; document.querySelector("#result-distance").textContent = distanceText; document.querySelector("#result-address").textContent = "지역 확인 중..."; const riskAnalysisPromise = requestRiskAssessment(sourceLatitude, sourceLongitude); const predictedAddressPromise = reverseGeocode(latitude, longitude); const [predictedAddress] = await Promise.all([predictedAddressPromise, riskAnalysisPromise, wait(Math.max(0, LOADING_DURATION_MS - (Date.now() - startedAt)))]); showScreen("result"); document.querySelector("#result-address").textContent = predictedAddress; updateMap(sourceLatitude, sourceLongitude, latitude, longitude, predictedAddress); } catch (error) { console.error(error); alert(error.message); showScreen("search"); } finally { predictButton.disabled = false; } }
function fillSample() { document.querySelector("#age").value = 70; document.querySelector("#height_cm").value = 165; document.querySelector("#weight_kg").value = 62; document.querySelector("#temperature_c").value = 24; document.querySelector("#last_seen_latitude").value = 37.5665; document.querySelector("#last_seen_longitude").value = 126.978; document.querySelector("#missing_datetime").value = "2024-06-15T14:30"; placeInput.value = "서울특별시청"; state.selectedPlaceName = "서울특별시청"; setPlaceStatus("success", "예시 좌표가 설정되었습니다."); state.preprocessing.categorical.columns.forEach((column, index) => setSelectValue(fieldIds[column], state.preprocessing.categorical.categories[index][0])); }
placeInput.addEventListener("input", () => { state.selectedPlaceName = ""; document.querySelector("#last_seen_latitude").value = ""; document.querySelector("#last_seen_longitude").value = ""; placeSearchResults.hidden = true; setPlaceStatus("", "장소 검색 버튼을 눌러 좌표를 설정해 주세요."); });
placeInput.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); searchPlace(); } });
form.addEventListener("submit", renderCoreInputSummary); form.addEventListener("submit", predict); placeSearchButton.addEventListener("click", searchPlace); document.querySelector("#sample-button").addEventListener("click", fillSample); document.querySelector("#back-to-search").addEventListener("click", () => showScreen("search")); window.addEventListener("DOMContentLoaded", loadAssets);
