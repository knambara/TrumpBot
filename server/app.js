const express = require("express");
const bodyParser = require("body-parser");
const compression = require("compression");
const path = require("path");

const scriptPath = path.join(__dirname, '../model/interact.py');
const {PythonShell} = require('python-shell', {
  mode: 'text',
  args: ['-q'],
  scriptPath
});

const pyShell = new PythonShell(scriptPath);
console.log(`[✔] connect to TrumpBot at ${scriptPath}`);

// Create Express server
const app = express();
app.set("port", process.env.PORT || 3000);
app.use(compression());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.use(function(request, response, next) {
  console.log(`received request at ${request.originalUrl}`);
  next();
});

app.get("/trumpbot", function (request, response) {
  const prompt = request.body.prompt;

  if (!prompt || typeof prompt !== 'string') {
    return response.status(400).send('Prompt string required');
  }

  if (prompt === '') {
    return response.status(400).send('Prompt cannot be an empty string');
  }

  const nwords = prompt.split(' ').length;
  if (nwords >= 50) {
    return response.status(400).send('Prompt is too long. Try a prompt less than 50 words');
  }

  pyShell.prependOnceListener('message', function (answer) {
    console.log(`[✔] Received answer (${answer}) for prompt (${prompt})`);
    return response.status(200).json({ prompt, answer });
  });
  pyShell.send(prompt);
});

// handle missing pages
app.get("*", function(request, response) {
  response.sendStatus(404);
});

// Handle ^C
process.on('SIGINT', () => {
  pyShell.end(function (err, code, signal) {
    if (err) throw err;
    console.log(`[i] Python shell exited with ${code} from ${signal}`);
    process.exit(0);
  });
});

module.exports = app;
