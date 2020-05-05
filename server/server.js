const app = require('./app.js');
const ip = require("ip");

/**
 * Start Express server.
 */
const server = app.listen(app.get("port"), () => {
      console.log(`[✔] App is running at http://${ip.address()}:${app.get("port")} in ${app.get("env")} mode`);
      console.log("[i] Press CTRL-C to stop\n");
  });

module.exports = server;