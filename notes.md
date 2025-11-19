---
layout: page
title: 学习总结
permalink: /notes/
---

这里会集中展示所有学习总结页面。

{%- assign notes_pages = site.pages | where_exp: "p", "p.path contains 'docs/'" | sort: "title" -%}

{%- if notes_pages.size == 0 -%}
<p>当前还没有学习总结文档，可以先在 <code>docs/</code> 中添加一些 Markdown 文件。</p>
{%- else -%}
<ul>
  {%- for p in notes_pages -%}
  <li>
    <a href="{{ p.url | relative_url }}">{{ p.title }}</a>
  </li>
  {%- endfor -%}
</ul>
{%- endif -%}
