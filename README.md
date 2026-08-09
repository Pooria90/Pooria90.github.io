# Pooria Ashrafian

Personal website for Pooria Ashrafian, built with [al-folio](https://github.com/alshedivat/al-folio) and deployed with GitHub Pages.

## Local development

```bash
bundle install
npm ci
bundle exec jekyll serve
```

The site is available at `http://localhost:4000/` while the development server is running.

## Production build

```bash
npm run lint:prettier
bundle exec jekyll build
```
