# realm_task_3_3_xai_uc3

## General Task Description

Components developed in Task 3.3 aim to implement agnostic XAI techniques on top of AI models that are used for various tasks such as classification, segmentation or next value prediction.
We aim to implement **two** XAI techniques per Use Case - that would be selected dynamically from the Fuzzy system based on the user's input (sensitivity value coming from the RIANA dashboard), 
implement bias and fairness metrics (as agreed [here](https://maastrichtuniversity.sharepoint.com/:w:/r/sites/FSE-REALM/_layouts/15/Doc.aspx?sourcedoc=%7B9EDAE561-2787-42D1-BBB8-C9320C0B1F25%7D&file=Report%20on%20Bias%20and%20Fairness%20Metrics%20%5BTask%203.3%5D.docx&action=default&mobileredirect=true)) based on model
outputs and extract outputs in a digestible manner (images, metrics, etc).

This component, no matter the Use Case, expects as input:

* Sensitivity value (RIANA dashboard)
* Trained model (AI Orchestrator)
* Compatible dataset (AI Orchestrator)

This component, no matter the Use Case, returns as output:

* XAI methodology output (depending on the Use Case - image or json file)
* Fairness and Bias results (depending on the Use Case - nothing if we are talking for images or json file)

## 📜 License & Usage

All rights reserved by MetaMinds Innovations.
