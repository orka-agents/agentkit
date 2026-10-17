#!/opt/agentkit/bin/node

const readline = require("node:readline");

const input = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

input.once("line", (line) => {
  const request = JSON.parse(line);
  process.stdout.write(
    `${JSON.stringify({
      jsonrpc: "2.0",
      id: request.id,
      result: {
        protocolVersion: "2025-06-18",
        capabilities: {},
        serverInfo: {
          name: "agentkit-portable-node-tool",
          version: "0.0.0",
        },
      },
    })}\n`,
  );
  input.close();
});
