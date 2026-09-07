const LANDMARK_CATEGORY_CODES = ["AT4", "CT1", "SW8", "PO3"];
const MAX_LANDMARKS = 20;

function clearNearbyLandmarks() {
  state.landmarkSearchId += 1;
  const list = document.querySelector("#landmark-list");
  list.replaceChildren();
  const status = document.createElement("li");
  status.className = "landmark-status";
  status.textContent = "주변 랜드마크를 찾는 중...";
  list.append(status);
}

function searchLandmarkCategory(places, categoryCode, center) {
  return new Promise((resolve) => {
    let completed = false;
    const finish = (results) => {
      if (completed) return;
      completed = true;
      clearTimeout(timeoutId);
      resolve(results);
    };
    const timeoutId = setTimeout(() => finish([]), 6000);
    try {
      places.categorySearch(categoryCode, (results, status) => {
        finish(status === kakao.maps.services.Status.OK ? results : []);
      }, {
        location: center,
        radius: 1000,
        size: 15,
        sort: kakao.maps.services.SortBy.DISTANCE,
      });
    } catch (error) {
      console.warn(`랜드마크 카테고리 ${categoryCode} 검색 실패`, error);
      finish([]);
    }
  });
}

async function renderNearbyLandmarks(center) {
  clearNearbyLandmarks();
  const searchId = state.landmarkSearchId;
  const list = document.querySelector("#landmark-list");
  if (!kakao.maps.services?.Places) {
    list.firstElementChild.textContent = "랜드마크 검색 서비스를 사용할 수 없습니다.";
    return;
  }

  const places = new kakao.maps.services.Places();
  const categoryResults = await Promise.all(
    LANDMARK_CATEGORY_CODES.map((code) => searchLandmarkCategory(places, code, center))
  );
  if (searchId !== state.landmarkSearchId) return;

  const uniquePlaces = new Map();
  categoryResults.flat().forEach((place) => uniquePlaces.set(place.id, place));
  const nearbyPlaces = [...uniquePlaces.values()]
    .sort((a, b) => Number(a.distance) - Number(b.distance))
    .slice(0, MAX_LANDMARKS);

  list.replaceChildren();
  if (!nearbyPlaces.length) {
    const status = document.createElement("li");
    status.className = "landmark-status";
    status.textContent = "반경 1km 안에서 표시할 랜드마크를 찾지 못했습니다.";
    list.append(status);
    return;
  }

  nearbyPlaces.forEach((place) => {
    const item = document.createElement("li");
    const name = document.createElement("strong");
    const details = document.createElement("span");
    const category = String(place.category_name || "주변 장소").split(" > ").pop();
    const distance = Number(place.distance);
    name.textContent = place.place_name || "이름 없는 장소";
    details.textContent = [
      category,
      Number.isFinite(distance) ? `${distance.toLocaleString("ko-KR")}m` : "",
    ].filter(Boolean).join(" · ");
    item.append(name, details);
    list.append(item);
  });
}

// Keep the first map view anchored to the coordinates the user entered.
// A distant model estimate should not move the initial viewport away from it.
function updateMap(seenLatitude, seenLongitude, latitude, longitude, predictedAddress) {
  const container = document.querySelector("#map");
  const seenPosition = new kakao.maps.LatLng(seenLatitude, seenLongitude);
  const predictedPosition = new kakao.maps.LatLng(latitude, longitude);

  if (!state.map) {
    state.map = new kakao.maps.Map(container, { center: seenPosition, level: 5 });
  }

  [state.seenMarker, state.predictionMarker].forEach((marker) => marker?.setMap(null));
  state.predictionCircle?.setMap(null);
  state.routeLine?.setMap(null);

  state.seenMarker = new kakao.maps.Marker({ position: seenPosition, title: "Last seen" });
  state.predictionMarker = new kakao.maps.Marker({ position: predictedPosition, title: predictedAddress });
  state.predictionCircle = new kakao.maps.Circle({
    center: predictedPosition,
    radius: 1000,
    strokeWeight: 2,
    strokeColor: "#e53935",
    strokeOpacity: 0.85,
    strokeStyle: "solid",
    fillColor: "#e53935",
    fillOpacity: 0.16,
  });
  state.routeLine = new kakao.maps.Polyline({
    path: [seenPosition, predictedPosition],
    strokeWeight: 3,
    strokeColor: "#2f6fed",
    strokeOpacity: 0.65,
    strokeStyle: "shortdash",
  });

  state.seenMarker.setMap(state.map);
  state.predictionCircle.setMap(state.map);
  state.predictionMarker.setMap(state.map);
  state.routeLine.setMap(state.map);
  renderNearbyLandmarks(predictedPosition);
  const bounds = new kakao.maps.LatLngBounds();
  bounds.extend(seenPosition);
  bounds.extend(predictedPosition);
  // Include the full 1 km prediction radius when fitting the viewport.
  const latitudeDelta = 1 / 111.32;
  const longitudeDelta = 1 / (111.32 * Math.max(Math.cos(latitude * Math.PI / 180), 0.01));
  bounds.extend(new kakao.maps.LatLng(latitude + latitudeDelta, longitude));
  bounds.extend(new kakao.maps.LatLng(latitude - latitudeDelta, longitude));
  bounds.extend(new kakao.maps.LatLng(latitude, longitude + longitudeDelta));
  bounds.extend(new kakao.maps.LatLng(latitude, longitude - longitudeDelta));
  state.mapBounds = bounds;
  state.map.setBounds(bounds, 70, 70, 70, 70);
  if (state.map.getLevel() < 4) state.map.setLevel(4);
}
