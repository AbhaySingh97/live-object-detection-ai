const localtunnel = require('localtunnel');
const fs = require('fs');
const path = require('path');

const PORT = 5000;
let tunnelInstance = null;

async function startTunnel() {
  try {
    console.log(`[Tunnel] Opening public tunnel for port ${PORT}...`);
    tunnelInstance = await localtunnel({ port: PORT, subdomain: 'live-object-vision-api' });

    console.log(`[Tunnel] Live Public URL: ${tunnelInstance.url}`);
    
    fs.writeFileSync(path.join(__dirname, 'active_tunnel_url.txt'), tunnelInstance.url);

    tunnelInstance.on('close', () => {
      console.log('[Tunnel] Closed. Reconnecting in 3s...');
      setTimeout(startTunnel, 3000);
    });

    tunnelInstance.on('error', (err) => {
      console.error('[Tunnel] Error:', err.message);
      try { tunnelInstance.close(); } catch(e){}
    });
  } catch (err) {
    console.error('[Tunnel] Failed to start:', err.message);
    setTimeout(startTunnel, 5000);
  }
}

startTunnel();
