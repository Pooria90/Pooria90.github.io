---
layout: page
title: Notes
permalink: /notes/
description: Technical notes, practical lessons, general learnings, and life systems.
nav: true
nav_order: 4
---

This is where I collect technical notes, useful patterns, and lessons I want to return to. Topics will span:

- ML Systems
- LLM Inference
- Agentic Systems
- Biomedical Imaging
- C++ / Parallel Computing
- Career Notes
- Life Systems

{% if site.posts.size > 0 %}
## Recent notes

{% for post in site.posts %}
### [{{ post.title }}]({{ post.url | relative_url }})

{{ post.description | default: post.excerpt | strip_html | truncate: 180 }}

<small>{{ post.date | date: "%B %-d, %Y" }}{% if post.tags.size > 0 %} · {{ post.tags | join: ", " }}{% endif %}</small>
{% endfor %}
{% else %}
## Notes coming soon

I am setting up this section now. I will publish notes when they are useful enough to share—no filler posts in the meantime.
{% endif %}
