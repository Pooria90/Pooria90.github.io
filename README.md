# Pooria Ashrafian

Personal website for Pooria Ashrafian, built with [al-folio](https://github.com/alshedivat/al-folio) and deployed with GitHub Pages.

## Local development

The site requires Ruby, Bundler, and the native-extension build tools. On Ubuntu/Debian, install them with:

```bash
sudo apt-get update
sudo apt-get install ruby-full ruby-dev build-essential
```

Then install the Ruby dependencies inside the project and start Jekyll:

```bash
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll serve
```

The site is available at `http://localhost:4000/` while the development server is running.

If `bundle install` reports `mkmf.rb can't find header files for ruby`, the Ruby development headers are missing; install `ruby-dev` as shown above and rerun `bundle install`.

Node.js dependencies are not required to serve the site. Install them with `npm ci` when running formatting or visual tests.

## Production build

```bash
npm ci
npm run lint:prettier
bundle exec jekyll build
```
