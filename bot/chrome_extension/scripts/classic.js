// GeoGuessr Bot - Classic Mode
// Uses direct API submission (most reliable method)
const API_ENDPOINT = "http://127.0.0.1:5000/api/v1/predict";

(async () => {
  console.log("🤖 GeoGuessr Bot v4 - Classic Mode");
  
  if (!window.location.href.includes('/game/')) {
    console.log("❌ Not on a game page");
    return;
  }

  // Add status overlay
  addStatusOverlay();
  
  // Get game state from the page
  let gameState = await getGameState();
  console.log("📊 Initial game state:", gameState);
  
  // Main loop
  while (true) {
    const round = gameState?.currentRound || 1;
    setStatus(`Round ${round}: Starting...`, 'blue');
    
    // Step 1: Wait for round to be playable
    console.log(`\n===== ROUND ${round} =====`);
    const ready = await waitForPlayableRound();
    if (!ready) {
      console.log("⚠️ Could not detect playable round, retrying...");
      await sleep(2000);
      continue;
    }
    
    // Step 2: Wait for panorama to load
    setStatus(`Round ${round}: Loading panorama...`, 'blue');
    await sleep(2500);
    
    // Step 3: Take screenshot
    setStatus(`Round ${round}: Capturing...`, 'blue');
    console.log("📸 Taking screenshot...");
    hideOverlays(true);
    await sleep(400);
    const imageData = await takeScreenshot();
    hideOverlays(false);
    
    if (!imageData) {
      console.error("❌ Screenshot failed");
      await sleep(2000);
      continue;
    }
    
    // Step 4: Get ML prediction
    setStatus(`Round ${round}: Getting prediction...`, 'blue');
    console.log("🔮 Sending to ML model...");
    
    let lat, lng;
    try {
      const response = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const result = await response.json();
      lat = result.results.lat;
      lng = result.results.lng;
      console.log(`🎯 Prediction: ${lat.toFixed(4)}, ${lng.toFixed(4)}`);
    } catch (e) {
      console.error("❌ ML API error:", e);
      setStatus("API Error!", 'red');
      await sleep(3000);
      continue;
    }
    
    setStatus(`Round ${round}: ${lat.toFixed(2)}, ${lng.toFixed(2)}`, 'green');
    
    // Step 5: Submit guess via GeoGuessr API
    console.log("📤 Submitting guess...");
    const submitResult = await submitGuess(lat, lng, round);
    
    if (submitResult.success) {
      console.log("✅ Guess submitted successfully!");
      console.log("📊 Score:", submitResult.data?.roundScore?.amount || 'N/A');
      console.log("📏 Distance:", submitResult.data?.roundScore?.distance || 'N/A');
      
      // Update game state from response
      if (submitResult.data) {
        gameState = {
          currentRound: (submitResult.data.currentRoundNumber || round) + 1,
          totalRounds: submitResult.data.bounds?.max || 5,
        };
      }
    } else {
      console.log("⚠️ Guess submission issue:", submitResult.error);
    }
    
    // Step 6: Wait for round to end and progress
    setStatus(`Round ${round}: Waiting for next round...`, 'blue');
    await waitForRoundTransition();
    
    console.log("➡️ Moving to next round...");
    await sleep(1000);
  }
})();

// ================== FUNCTIONS ==================

function addStatusOverlay() {
  const overlay = document.createElement('div');
  overlay.id = 'geobot-overlay';
  overlay.innerHTML = `
    <div id="geobot-status" style="
      position: fixed; top: 15px; right: 15px; z-index: 999999;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white; padding: 12px 18px; border-radius: 10px;
      font-family: 'Segoe UI', system-ui, sans-serif;
      font-size: 14px; font-weight: 600;
      box-shadow: 0 4px 15px rgba(0,0,0,0.3);
      transition: all 0.3s ease;
    ">🤖 Bot Active</div>
  `;
  document.body.appendChild(overlay);
}

function setStatus(text, color = 'blue') {
  const el = document.getElementById('geobot-status');
  if (!el) return;
  
  el.textContent = `🤖 ${text}`;
  
  const gradients = {
    blue: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    green: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
    red: 'linear-gradient(135deg, #eb3349 0%, #f45c43 100%)',
    orange: 'linear-gradient(135deg, #f7971e 0%, #ffd200 100%)',
  };
  el.style.background = gradients[color] || gradients.blue;
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function takeScreenshot() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ action: "screenshot" }, (response) => {
      resolve(response?.image || null);
    });
  });
}

async function getGameState() {
  // Try to get game state from URL or page
  const gameId = getGameId();
  if (!gameId) return null;
  
  try {
    const response = await fetch(`https://www.geoguessr.com/api/v3/games/${gameId}`, {
      credentials: 'include'
    });
    if (response.ok) {
      const data = await response.json();
      return {
        currentRound: data.round || 1,
        totalRounds: data.roundCount || 5,
        token: data.token,
      };
    }
  } catch (e) {
    console.log("Could not fetch game state:", e);
  }
  
  return { currentRound: 1, totalRounds: 5 };
}

function getGameId() {
  const match = window.location.href.match(/\/game\/([A-Za-z0-9]+)/);
  return match ? match[1] : null;
}

async function submitGuess(lat, lng, roundNumber) {
  const gameId = getGameId();
  if (!gameId) {
    return { success: false, error: "No game ID" };
  }
  
  const url = `https://game-server.geoguessr.com/api/game/${gameId}/guess`;
  
  console.log(`📡 POST ${url}`);
  console.log(`📍 Payload: { lat: ${lat}, lng: ${lng}, roundNumber: ${roundNumber} }`);
  
  try {
    const response = await fetch(url, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-client": "web",
        "Origin": "https://www.geoguessr.com",
        "Referer": `https://www.geoguessr.com/game/${gameId}`,
      },
      body: JSON.stringify({
        lat: lat,
        lng: lng,
        roundNumber: roundNumber,
      }),
    });
    
    console.log(`📊 Response status: ${response.status}`);
    console.log(`📊 Response headers:`, Object.fromEntries(response.headers.entries()));
    
    const responseText = await response.text();
    console.log(`📊 Response body (raw):`, responseText);
    
    let data;
    try {
      data = JSON.parse(responseText);
    } catch (e) {
      console.log("⚠️ Response is not JSON");
      data = { raw: responseText };
    }
    
    if (response.ok && data.roundScore) {
      console.log("✅ GUESS ACCEPTED!");
      console.log(`   Score: ${data.roundScore.amount} points`);
      console.log(`   Distance: ${data.roundScore.distance} meters`);
      return { success: true, data: data };
    } else if (response.ok) {
      console.log("⚠️ Response OK but no roundScore - guess may not have registered");
      return { success: true, data: data };
    } else {
      console.log(`❌ GUESS REJECTED: ${response.status}`);
      return { success: false, error: data.message || `HTTP ${response.status}`, data: data };
    }
  } catch (e) {
    console.error("❌ Submit error:", e);
    return { success: false, error: e.message };
  }
}

async function waitForPlayableRound(timeout = 30000) {
  const start = Date.now();
  
  while (Date.now() - start < timeout) {
    // Check for panorama (indicates round is active)
    const hasPanorama = document.querySelector('[class*="panorama"]') || 
                        document.querySelector('.gmnoprint') ||
                        document.querySelector('[data-qa="panorama"]');
    
    // Check for guess button
    const hasGuessButton = document.querySelector('[data-qa="perform-guess"]') ||
                           document.querySelector('.guess-map__guess-button') ||
                           document.querySelector('button[class*="guess"]');
    
    // Check we're not on results screen
    const onResults = document.querySelector('[class*="round-result"]') ||
                      document.querySelector('[data-qa="round-result"]');
    
    if (hasPanorama && hasGuessButton && !onResults) {
      console.log("✅ Round is playable");
      return true;
    }
    
    // If on results, try to click through
    if (onResults) {
      console.log("📊 On results screen, clicking through...");
      await clickThroughResults();
    }
    
    await sleep(300);
  }
  
  console.log("⚠️ Timeout waiting for playable round");
  return false;
}

async function waitForRoundTransition(timeout = 15000) {
  const start = Date.now();
  
  // First wait for the results screen to appear
  while (Date.now() - start < timeout) {
    const resultsVisible = document.querySelector('[class*="round-result"]') ||
                           document.querySelector('[data-qa="round-result"]') ||
                           document.querySelector('[class*="result-layout"]');
    
    // Or check if guess button disappeared
    const guessButton = document.querySelector('[data-qa="perform-guess"]');
    
    if (resultsVisible || !guessButton) {
      console.log("📊 Results screen detected");
      await sleep(1500); // Let animations play
      await clickThroughResults();
      return;
    }
    
    await sleep(200);
  }
  
  console.log("⚠️ Timeout waiting for round transition, trying to continue anyway...");
  await clickThroughResults();
}

async function clickThroughResults() {
  // Try various continue buttons
  const buttonSelectors = [
    '[data-qa="close-round-result"]',
    '[data-qa="play-next-round"]',
    'button[class*="next"]',
    'button[class*="continue"]',
    '[class*="result"] button',
    '[class*="round-result"] button',
    'button[class*="close"]',
  ];
  
  for (const selector of buttonSelectors) {
    const btn = document.querySelector(selector);
    if (btn && isVisible(btn)) {
      console.log(`🖱️ Clicking: ${selector}`);
      btn.click();
      await sleep(500);
      
      // Check if we need to click again
      const stillVisible = document.querySelector(selector);
      if (stillVisible && isVisible(stillVisible)) {
        stillVisible.click();
      }
      
      return;
    }
  }
  
  // Fallback: press Space bar (works on GeoGuessr results screen)
  console.log("⌨️ Pressing Space");
  const event = new KeyboardEvent('keydown', {
    key: ' ',
    code: 'Space',
    keyCode: 32,
    which: 32,
    bubbles: true,
    cancelable: true,
  });
  document.dispatchEvent(event);
  document.body.dispatchEvent(event);
  
  await sleep(300);
  
  // Also try clicking the page body
  document.body.click();
}

function isVisible(el) {
  if (!el) return false;
  const style = window.getComputedStyle(el);
  return style.display !== 'none' && 
         style.visibility !== 'hidden' && 
         style.opacity !== '0' &&
         el.offsetParent !== null;
}

function hideOverlays(hide) {
  const display = hide ? 'none' : '';
  
  // Hide bot overlay
  const botOverlay = document.getElementById('geobot-overlay');
  if (botOverlay) botOverlay.style.display = display;
  
  // Hide game UI elements for clean screenshot
  const selectors = [
    '.gmnoprint',
    '[class^="game-panorama_controls"]',
    '[class^="game_controls"]', 
    '[class^="game_guess"]',
    '[class^="game-map"]',
    '[class^="game_hud"]',
    '[class*="compass"]',
    '[class*="ad-"]',
  ];
  
  selectors.forEach(sel => {
    document.querySelectorAll(sel).forEach(el => {
      el.style.display = display;
    });
  });
  
  // Also hide SVG paths (Google Maps UI)
  document.querySelectorAll('path').forEach(p => p.style.display = display);
}
