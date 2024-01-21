---
layout: page
title: Publications
permalink: /publications/
---

<style>
    .pub-card {
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        width: 100%;
        border-radius: 5px;
        display: block;
        margin: auto;
        background-color: #303030;
    }

    .pub-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.4);
    }

    .pub-card img {
        width: 100%;
        border-radius: 5px 5px 0 0;
    }

    .pub-card h2 {
        margin-top: 0.2em;
        margin-bottom: 0;
        line-height: 1.0;
    }

    .pub-card h4 {
        margin-top: 0.5em;
        margin-bottom: 0.2em;
        line-height: 0.9;
    }

    .pub-card h5 {
        margin-top: 0.2em;
        margin-bottom: 0.5em;
    }

    .pub-card p {
        line-height: 1.2;
    }

    .pub-card button {
    background-color: #D27D2D;
    border: none;
    color: white;
    padding: 5px 5px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 4px;
    font-family: inherit;
    }

    .pub-card button:hover {
    background-color: #FFAA33; /* Slightly darker orange when hovering */
    }

    .pub-card button:disabled {
    background-color: #888; /* Grey background for disabled buttons */
    cursor: not-allowed; /* Cursor indicating the button is disabled */
    }

</style>

<div class="pub-card">
    <img src="https://cdn.gamma.app/8sbvs4icoang31g/e07b9c6ebc564bf5ad924115b7eb9c13/original/Screenshot-2023-09-24-170020.png" alt="Image">
    <div style="padding: 2px 16px;">
        <h2>Exploring the Synergy Between Vision-Language
Pretraining and ChatGPT for Artwork
Captioning: A Preliminary Study</h2>
        <h4>Giovanna Castellano, Nicola Fanelli, Raffaele Scaringi, Gennaro Vessio</h4>
        <h5>FAPER Workshop - ICIAP 2023 (In press)</h5>
        <p>This paper explores the complex task of generating textual descriptions for the images of artworks with neural networks. A novel synthetic dataset of captions for the images in ArtGraph/WikiArt is collected using ChatGPT and is refined with CLIP. A VLP model (GIT) and a vision transformer are fine-tuned using instance weighting and multi-task learning to generate rich and expressive artwork descriptions.</p>
        <a href="https://link.springer.com/chapter/10.1007/978-3-031-51026-7_27">
        <button>Paper</button>
        </a>
    <a href="https://www.researchgate.net/publication/377558832_Exploring_the_Synergy_Between_Vision-Language_Pretraining_and_ChatGPT_for_Artwork_Captioning_A_Preliminary_Study">
    <button>ResearchGate</button>
    </a>
    <a href="https://paperswithcode.com/paper/exploring-the-synergy-between-vision-language">
    <button>Papers with Code</button>
    </a>
    <a href="https://github.com/nicolafan/neural-artwork-caption-generator">
    <button>GitHub</button>
    </a>
    </div>
</div>

