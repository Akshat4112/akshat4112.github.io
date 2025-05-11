# Personal Portfolio Website

This repository contains the source code for my personal portfolio website hosted at [akshat4112.github.io](https://akshat4112.github.io/).

## Overview

This is a static website built with [Hugo](https://gohugo.io/) using the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme. The site serves as a professional portfolio showcasing my work, publications, talks, and expertise in Machine Learning Engineering, Generative AI, Diffusion Models, and LLMs.

## Technology Stack

- **Static Site Generator**: [Hugo](https://gohugo.io/)
- **Theme**: [PaperMod](https://github.com/adityatelange/hugo-PaperMod)
- **Hosting**: GitHub Pages
- **CI/CD**: GitHub Actions

## Site Structure

- **Posts**: Technical blog posts and articles
- **Publications**: Research papers and technical publications
- **Talks**: Conference presentations and speaking engagements
- **Events**: Events I've participated in or organized
- **About**: Professional information and bio
- **CV**: Link to my curriculum vitae

## Local Development

### Prerequisites

- [Hugo Extended](https://gohugo.io/installation/) (v0.147.2 or later)
- Git

### Setup and Run

1. Clone the repository:
   ```bash
   git clone https://github.com/akshat4112/akshat4112.github.io.git
   cd akshat4112.github.io
   ```

2. Initialize and update submodules:
   ```bash
   git submodule update --init --recursive
   ```

3. Start the local development server:
   ```bash
   hugo server --bind 0.0.0.0 --baseURL http://localhost:1313 --disableFastRender
   ```

4. View the site at [http://localhost:1313](http://localhost:1313)

### Adding Content

#### Creating a new post:
```bash
hugo new posts/my-new-post.md
```

#### Creating other content types:
```bash
hugo new publications/my-publication.md
hugo new talks/my-talk.md
hugo new events/my-event.md
```

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch, using the GitHub Actions workflow defined in `.github/workflows/hugo.yml`.

## Customization

- **Site Configuration**: Edit `config.yml` to modify site settings, menus, and social links
- **Theme Customization**: Override theme templates by creating matching files in the `layouts/` directory
- **Styling**: Customize CSS by adding files to `assets/css/`

## License

The content of this project is licensed under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/), and the underlying source code is licensed under the [MIT license](https://opensource.org/licenses/mit-license.php).