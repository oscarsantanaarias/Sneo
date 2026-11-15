import { readFile, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { paths } from './config.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const configPath = path.join(paths.dataDir, 'plugins.json');

let loaded = false;
let pluginTools = [];
const connections = new Map();

const readConfig = async () => {
  if(!existsSync(configPath)){
    await writeFile(configPath, JSON.stringify({servers: []}, null, 2));
    return {servers: []};
  }

  try {
    const raw = await readFile(configPath, 'utf8');
    const data = JSON.parse(raw);
    return {servers: Array.isArray(data.servers) ? data.servers : []};
  } catch(e) {
    return {servers: []};
  }
};

const connectServer = async (server) => {
  const name = String(server.name || '').trim();

  if(!name || !server.command || server.enabled === false){
    return;
  }

  const transport = new StdioClientTransport({
    command: String(server.command),
    args: Array.isArray(server.args) ? server.args.map(String) : [],
    env: {...process.env, ...(server.env || {})}
  });

  const client = new Client({name: 'sneo-chat', version: '1.0.0'}, {capabilities: {}});
  await client.connect(transport);

  const listed = await client.listTools();
  connections.set(name, client);

  for(const tool of listed.tools || []){
    pluginTools.push({
      type: 'function',
      function: {
        name: `plugin__${name}__${tool.name}`,
        description: `[plugin ${name}] ${tool.description || tool.name}`,
        parameters: tool.inputSchema && typeof tool.inputSchema === 'object'
          ? tool.inputSchema
          : {type: 'object', properties: {}}
      }
    });
  }
};

const connectAll = async () => {
  pluginTools = [];
  const {servers} = await readConfig();

  for(const server of servers){
    try {
      await connectServer(server);
    } catch(e) {
      console.error(`[plugins] no pude conectar "${server?.name}": ${e.message}`);
    }
  }
};

export const ensurePluginsLoaded = async () => {
  if(loaded){
    return;
  }

  loaded = true;

  try {
    await connectAll();
  } catch(e) {
    console.error(`[plugins] carga fallo: ${e.message}`);
  }
};

export const getPluginTools = () => pluginTools;

export const isPluginTool = (name) => typeof name === 'string' && name.startsWith('plugin__');

export const runPluginTool = async (name, args) => {
  const rest = String(name).slice('plugin__'.length);
  const sep = rest.indexOf('__');

  if(sep < 0){
    return {error: `Nombre de plugin invalido: ${name}`};
  }

  const server = rest.slice(0, sep);
  const tool = rest.slice(sep + 2);
  const client = connections.get(server);

  if(!client){
    return {error: `Plugin "${server}" no esta conectado. Revisa data/plugins.json o usa plugin_reload.`};
  }

  const cleanArgs = {...(args || {})};
  delete cleanArgs.project;

  try {
    const result = await client.callTool({name: tool, arguments: cleanArgs});
    const text = Array.isArray(result.content)
      ? result.content.map(part => part.type === 'text' ? part.text : JSON.stringify(part)).join('\n')
      : JSON.stringify(result.content ?? result);

    return {plugin: server, tool, isError: result.isError || false, output: text.slice(0, 12000)};
  } catch(e) {
    return {error: `plugin ${server}/${tool} fallo: ${e.message}`};
  }
};

export const reloadPlugins = async () => {
  for(const client of connections.values()){
    try {
      await client.close();
    } catch(e) {}
  }

  connections.clear();
  await connectAll();

  return {reloaded: true, servers: [...connections.keys()], tools: pluginTools.length};
};

export const listPlugins = async () => {
  const {servers} = await readConfig();

  return {
    configured: servers.map(item => ({name: item.name, command: item.command, enabled: item.enabled !== false})),
    connected: [...connections.keys()],
    tools: pluginTools.map(item => item.function.name)
  };
};
