module.exports = {
  apps : [{
    name: 'TrumpBotServer',
    script: 'server.js',

    // Options reference: https://pm2.keymetrics.io/docs/usage/application-declaration/
    instances: 0,
    autorestart: true,
    watch: false,

    max_memory_restart: '1G',
    ignore_watch : ["node_modules", "[\/\\]\./", "example_responses", "logs", "README", "package.json", "package-lock.json", "*config*"],
    env: {
      NODE_ENV: 'development'
    },
    env_production: {
      NODE_ENV: 'production'
    }
  }],

  deploy : {
    production : {
      user : 'node',
      host : '212.83.163.1',
      ref  : 'origin/master',
      repo : 'git@github.com:repo.git',
      path : '/var/www/production',
      'post-deploy' : 'npm install && pm2 reload ecosystem.config.js --env production'
    }
  }
};
