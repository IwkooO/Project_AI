(async () => {
  console.log("🤖 GeoGuessr Bot Started - Classic Mode");
  console.log("📍 API Endpoint: http://127.0.0.1:5000/api/v1/predict");
  console.log("🎮 Waiting for game to start...");

  // Add visual indicator
  const botIndicator = document.createElement('div');
  botIndicator.id = 'geoguessr-bot-indicator';
  botIndicator.innerHTML = '🤖 Bot Active';
  botIndicator.style.cssText = `
    position: fixed;
    top: 10px;
    right: 10px;
    background: rgba(0, 123, 255, 0.9);
    color: white;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: bold;
    z-index: 10000;
    pointer-events: none;
  `;
  document.body.appendChild(botIndicator);

  let currentRoundNumber = 1;

  while (true) {
    console.log(`🔄 Round ${currentRoundNumber}: Waiting for guess button...`);
    await waitTillAppears(".guess-map__guess-button");
    console.log(`✅ Round ${currentRoundNumber}: Guess button found, starting prediction...`);
    await wait(1000);

    console.log("📸 Hiding GUI and capturing screenshot...");
    hideGUI(true);
    const response = await screenshot();
    const image = response.image;
    console.log("📸 Screenshot captured, showing GUI...");
    hideGUI(false);
    await wait(250);

    // Call API for prediction
    console.log("🔮 Sending image to ML model...");
    const startTime = Date.now();
    const apiResp = await fetch("http://127.0.0.1:5000/api/v1/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: image,
      }),
    });

    if (!apiResp.ok) {
      console.error("❌ API Error:", apiResp.status, apiResp.statusText);
      await wait(5000); // Wait before retrying
      continue;
    }

    const guess = await apiResp.json();
    const predictionTime = Date.now() - startTime;

    console.log(`🎯 Round ${currentRoundNumber}: Prediction received in ${predictionTime}ms`);
    console.log(`📍 Predicted Location: ${guess.results.lat.toFixed(4)}, ${guess.results.lng.toFixed(4)}`);
    console.log("📤 Submitting guess to GeoGuessr...");

    let result;
    let retryCount = 0;
    do {
      result = await submitGuessClassic(guess.results.lat, guess.results.lng, currentRoundNumber);
      console.log(`📊 Round ${currentRoundNumber}: Guess submitted - Status: ${result.resp.status}`);

      if (result.resp.status == 400) {
        retryCount++;
        console.log(`⚠️  Round ${currentRoundNumber}: Submission failed (attempt ${retryCount}), retrying...`);
        await wait(1000);
      } else {
        console.log(`✅ Round ${currentRoundNumber}: Guess accepted!`);
      }

      if (!result.body.currentRoundNumber) {
        currentRoundNumber += 1;
        console.log(`🔄 Moving to Round ${currentRoundNumber}`);
      } else {
        currentRoundNumber = result.body.currentRoundNumber + 1;
        console.log(`🔄 Server indicates Round ${currentRoundNumber}`);
      }
    } while (result.resp.status == 400 && retryCount < 3);

    if (result.resp.status == 400) {
      console.error(`❌ Round ${currentRoundNumber}: Failed to submit guess after 3 attempts`);
    }

    console.log(`⏳ Round ${currentRoundNumber}: Waiting for next round...`);
    await waitTillDisappears(".guess-map__guess-button");
    console.log(`🎮 Round ${currentRoundNumber}: Round complete, waiting for next round...`);
  }
})();

function screenshot() {
  console.log("📸 Requesting screenshot from background script...");
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(
      {
        action: "screenshot",
      },
      (response) => {
        console.log("📸 Screenshot received from background script");
        resolve(response);
      }
    );
  });
}

function getGameID() {
  const urlSplit = window.location.href.split("/");
  const gameID = urlSplit[urlSplit.length - 1];
  return gameID;
}

async function submitGuessClassic(lat, lng, roundNumber) {
  const gameID = getGameID();
  const apiURL = "https://game-server.geoguessr.com/api/game/" + gameID + "/guess";

  console.log(`📤 Round ${roundNumber}: Submitting guess to ${apiURL}`);
  console.log(`📍 Coordinates: ${lat.toFixed(6)}, ${lng.toFixed(6)}`);

  const payload = {
    lat: lat,
    lng: lng,
    roundNumber: roundNumber,
  };

  const headers = {
    origin: "https://www.geoguessr.com",
    referer: apiURL,
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
    "sec-ch-ua":
      '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "macOS",
    "user-agent":
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
    "x-client": "web",
    "Content-Type": "application/json",
  };

  const resp = await fetch(apiURL, {
    method: "POST",
    credentials: "include",
    headers: headers,
    body: JSON.stringify(payload),
  });

  const body = await resp.json();
  console.log(`📊 Round ${roundNumber}: GeoGuessr response:`, body);

  return { resp, body };
}

async function wait(millis) {
  await new Promise((r) => setTimeout(r, millis));
}

async function waitTillAppears(selector) {
  console.log(`⏳ Waiting for element: ${selector}`);
  let attempts = 0;
  while (!document.querySelector(selector)) {
    await new Promise((r) => setTimeout(r, 100));
    attempts++;
    if (attempts % 50 === 0) { // Log every 5 seconds
      console.log(`⏳ Still waiting for: ${selector} (${attempts * 0.1}s)`);
      checkPanoramaStatus();
    }
  }
  console.log(`✅ Element found: ${selector}`);
}

function checkPanoramaStatus() {
  // Check if panorama container exists
  const panoramaContainer = document.querySelector('[data-qa="panorama"]') ||
                           document.querySelector('.panorama-container') ||
                           document.querySelector('[class*="panorama"]');

  if (!panoramaContainer) {
    console.log(`⚠️  No panorama container found - page might not be fully loaded`);
    return;
  }

  // Check for loading indicators
  const loadingElements = document.querySelectorAll('[class*="loading"], [class*="spinner"], [class*="progress"]');
  if (loadingElements.length > 0) {
    console.log(`⏳ Panorama still loading (${loadingElements.length} loading indicators found)`);
  } else {
    console.log(`✅ Panorama container found, appears loaded`);
  }

  // Check for Street View tiles in network (this is harder to detect from DOM)
  console.log(`💡 Tip: If bot doesn't start, try refreshing the page to reload panorama tiles`);
}

async function waitTillDisappears(selector) {
  console.log(`⏳ Waiting for element to disappear: ${selector}`);
  while (document.querySelector(selector)) {
    await new Promise((r) => setTimeout(r, 100));
  }
  console.log(`✅ Element disappeared: ${selector}`);
}

function log(content) {
  console.log("🤖 GeoGuessr Bot:", content);
  chrome.runtime.sendMessage(
    { action: "log", content: content },
    function (response) {}
  );
}

function hideGUI(hide) {
  const view = hide ? "none" : "";
  let paths = document.getElementsByTagName("path");
  for (let i = 0; i < paths.length; i++) {
    paths[i].style.display = view;
  }

  let mentions = document.getElementsByClassName("gmnoprint");
  for (let i = 0; i < mentions.length; i++) {
    mentions[i].style.display = view;
  }

  let controls = document.querySelectorAll("[class^=game-panorama_controls]");
  for (let i = 0; i < controls.length; i++) {
    controls[i].style.display = view;
  }

  let o_controls = document.querySelectorAll("[class^=game_controls]");
  for (let i = 0; i < o_controls.length; i++) {
    o_controls[i].style.display = view;
  }

  let o_map = document.querySelectorAll("[class^=game_guess]");
  for (let i = 0; i < o_map.length; i++) {
    o_map[i].style.display = view;
  }

  let guessMap = document.querySelectorAll("[class^=game-map]");
  for (let i = 0; i < guessMap.length; i++) {
    guessMap[i].style.display = view;
  }

  let chat = document.querySelectorAll("[class^=chat-input]");
  for (let i = 0; i < chat.length; i++) {
    chat[i].style.display = view;
  }

  let msg = document.querySelectorAll("[class^=chat-message]");
  for (let i = 0; i < msg.length; i++) {
    msg[i].style.display = view;
  }

  let hud = document.querySelectorAll("[class^=game_hud]");
  for (let i = 0; i < hud.length; i++) {
    hud[i].style.display = view;
  }

  let consent = document.getElementById("adconsent-usp-link");
  if (consent) {
    consent.style.display = view;
  }
}

function debugBase64(base64URL) {
  let win = window.open();
  win.document.write(
    '<iframe src="' +
      base64URL +
      '" frameborder="0" style="border:0; top:0px; left:0px; bottom:0px; right:0px; width:100%; height:100%;" allowfullscreen></iframe>'
  );
}
