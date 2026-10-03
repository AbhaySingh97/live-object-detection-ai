const localtunnel = require('localtunnel');
const fs = require('fs');
const path = require('path');

const PORT = 5000;
let tunnelInstance = null;

async function startTunnel() {
  try {
    console.log(`[Tunnel] Opening public tunnel for port ${PORT}...`);
    tunnelInstance = await localtunnel({ port: PORT });

    console.log(`[Tunnel] Live Public URL: ${tunnelInstance.url}`);
    
    // Save tunnel URL to file for client/services to read
    const urlFile = path.join(__dirname, '..', 'active_tunnel_url.txt');
    fs.writeFileSync(urlFile, tunnelInstance.url);

    tunnelInstance.on('close', () => {
      console.log('[Tunnel] Tunnel closed. Reconnecting in 3s...');
      setTimeout(startTunnel, 3000);
    });

    tunnelInstance.on('error', (err) => {
      console.error('[Tunnel] Error:', err.message);
      tunnelInstance.close();
    });
  } catch (err) {
    console.error('[Tunnel] Failed to start:', err.message);
    setTimeout(startTunnel, 5000);
  }
}

startTunnel();
