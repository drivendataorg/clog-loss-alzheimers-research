[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://www.drivendata.co/images/stallcatchers-vessels.jpg)

# Clog Loss: Advance Alzheimer’s Research with Stall Catchers

## Goal of the Competition

5.8 million Americans [live with Alzheimer’s dementia](https://www.alz.org/alzheimers-dementia/facts-figures), including 10% of all seniors 65 and older. Scientists at Cornell have discovered links between “stalls,” or clogged blood vessels in the brain, and Alzheimer’s. Stalls can reduce overall blood flow in the brain by 30%. The ability to prevent or remove stalls may transform how Alzheimer’s disease is treated.

[Stall Catchers](https://stallcatchers.com/main) is a citizen science project that crowdsources the analysis of Alzheimer’s disease research data provided by Cornell University’s Department of Biomedical Engineering. It resolves a pressing analytic bottleneck: for each hour of data collection it would take an entire week to analyze the results in the lab, which means an entire experimental dataset would take 6-12 months to analyze. Today, the Stall Catchers players are collectively analyzing data 5x faster than the lab while exceeding data quality requirements.

Through the Stall Catchers project, there is now a one-of-a-kind dataset that can be used to train and evaluate ML models on this task. Each exemplar is an image stack (a 3D image) taken from a live mouse brain showing blood vessels and blood flow. Each stack has an outline drawn around a target vessel segment and has been converted to an mp4 video file. **The objective of this competition is to classify the outlined vessel segment as flowing—if blood is moving through the vessel—or stalled if the vessel has no blood flow.**

## What's in this Repository

This repository contains code from winning competitors in the [Clog Loss: Advance Alzheimer’s Research with Stall Catchers](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
----- | ------------ | ---   | ---   | ---
1     | ZFTurbo      | 0.844 | 0.856 | <Description from the 1st place's writeup>
2     | kbrodt       | 0.846 | 0.838 | <Description from the 2nd place's writeup>
3     | LauraOnac    | 0.777 | 0.810 | <Description from the 3rd place's writeup>
Bonus | AZ_KA        | 0.467 | 0.536 | <Description from the 3rd place's writeup>

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: ["Advance Alzheimer’s Research with Stall Catchers - Benchmark"](https://www.drivendata.co/blog/stall-catchers-alzheimers-benchmark/)**
