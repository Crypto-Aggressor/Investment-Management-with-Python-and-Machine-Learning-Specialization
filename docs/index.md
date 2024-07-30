---
layout: home

title: "Investment Management with Python"
titleTemplate: "Empower Your Financial Strategy"

hero:
  name: "Investment Management with Python"
  text: "Master Financial Models & Portfolio Strategies"
  tagline: "Discover advanced portfolio management techniques and financial analytics powered by Python and Machine Learning."
  image:
    src: "./resources/cppi.jpg"
    alt: "Financial Models and Strategies"
    class: "hero-image"
  actions:
    - theme: brand
      text: "Explore Courses →"
      link: "/1-intro/1.0-overview"

features:
  - icon:
      src: "/resources/math-integral.svg"
    title: "Portfolio Management Fundamentals"
    details: "Dive deep into portfolio management fundamentals, exploring asset risks, returns, and optimization strategies."
  - icon:
      src: "/resources/bell-shaped-graph.svg"
    title: "Advanced Portfolio Analysis"
    details: "Leverage Python for advanced portfolio construction techniques, including robust covariance estimation and the Black-Litterman model."
  - icon:
      src: "/resources/cosine-graph.svg"
    title: "Asset-Liability Management"
    details: "Understand the intricacies of managing assets and liabilities to align with investment goals and risk tolerance."
---


<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: linear-gradient(120deg, #bd34fe 30%, #41d1ff);
  --vp-home-hero-image-background-image: linear-gradient(-45deg, #bd34fe 50%, #47caff 50%);
  --vp-home-hero-image-filter: blur(44px);
}

@media (min-width: 640px) {
  :root {
    --vp-home-hero-image-filter: blur(56px);
  }
}

@media (min-width: 960px) {
  :root {
    --vp-home-hero-image-filter: blur(68px);
  }
}

.custom-block.info {
  background-color: #3eaf7c; /* Brand color */
  color: white; /* Ensuring text is readable on the darker background */
  border-left: 5px solid darken(#3eaf7c, 10%); /* Slightly darker for the border */
  padding: 10px 20px; /* Padding inside the box */
  font-size: 20px; /* Increased font size */
  line-height: 1.5; /* Improved line height for better readability */
}

/* Adding custom styles for the hero image */
.hero-image {
  width: 200%; /* Adjust the width as needed */
  height: auto; /* Maintain aspect ratio */
  max-width: 1000px; /* Set a max-width for larger screens */
  display: block; /* Optional: ensures it is centered if also adding margin */
  margin: 0 auto; /* Optional: centers the image horizontally */
}
</style>

::: info Project Goals & Scope
After earning a specialization in **Investment Management with Python and Machine Learning** from EDHEC Business School ([**`Credential ID: WUTZABL42PW8`**](https://www.coursera.org/account/accomplishments/specialization/WUTZABL42PW8)), I compiled key financial and mathematical concepts into a Python module. This project is documented in Jupyter notebooks and here on Vitepress.

The main goal of this project is to provide an accessible, structured, and practical approach to modern investment strategies and portfolio management techniques. It aims to bridge the gap between academic theory and real-world application, enabling users—from students to professional investors—to implement and test various investment strategies using Python.

The content spans several core areas:

- **Fundamental Analysis**: Introducing basic concepts such as returns calculations and risk assessments.
- **Quantitative Methods**: Covering advanced statistical and mathematical techniques to optimize portfolios and manage risks effectively.
- **Machine Learning Applications**: Demonstrating how machine learning can be applied to enhance predictive accuracy in investment decisions.
- **Interactive Learning**: Each module includes interactive Jupyter notebooks that allow users to experiment with real data, tweak parameters, and observe the outcomes in real-time.

This educational toolkit is not just a passive learning resource but an active framework for engaging with and mastering the complexities of financial markets through technology.
:::
