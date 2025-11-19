---
layout: page
title: 美图欣赏
permalink: /gallery/
---

<div class="gallery-page">
  <h2>点击按钮展示图片</h2>
  <button class="gallery-btn" onclick="showImage()">Open</button>
  <img id="myImg" src="{{ '/ym.jpg' | relative_url }}" alt="Yang Mi">
</div>

<style>
  .gallery-page {
    text-align: center;
    margin-top: 2rem;
  }

  .gallery-btn {
    padding: 12px 28px;
    font-size: 18px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    background: #4caf50;
    color: white;
    transition: 0.3s;
    outline: none;
  }

  .gallery-btn:hover {
    background: #45a049;
    transform: scale(1.05);
  }

  .gallery-btn:active {
    transform: scale(0.95);
  }

  #myImg {
    width: 320px;
    max-width: 90%;
    border-radius: 12px;
    display: none;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    transition: all 0.4s ease;
    margin-top: 20px;
  }

  #myImg.show {
    display: block;
    animation: fadeIn 0.8s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
  }
</style>

<script>
  function showImage() {
    var img = document.getElementById('myImg');
    if (!img.classList.contains('show')) {
      img.classList.add('show');
    }
  }
</script>
