---
layout: page
title: 学习总结
permalink: /notes/
---

这里自动汇总 `docs/` 目录下的学习笔记。

> 只要你在仓库的 `docs/` 目录中添加新的 Markdown 文件，并推送到 GitHub，
> 这个页面就会自动出现相应的链接，无需手动修改导航。

<ul>
  {%- assign notes = site.static_files | where_exp: "file", "file.path contains '/docs/'" -%}
  {%- if notes.size == 0 -%}
    <li>当前还没有学习总结文档，可以先在 <code>docs/</code> 中添加一些 Markdown 文件。</li>
  {%- else -%}
    {%- for file in notes -%}
      <li>
        <a href="{{ file.path | relative_url }}">
          {{ file.name | replace: '.md', '' }}
        </a>
      </li>
    {%- endfor -%}
  {%- endif -%}
</ul>
