# NER PII Masking

This module is inspired by
LlamaIndex ['PII Masking'](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/PII/#option-1-use-ner-model-for-pii-masking).

Use a Hugging Face NER model for PII Masking

## What is PII Masking?

PII(Personally Identifiable Information) Masking is a data protection method that obscures sensitive personal
information to prevent unauthorized access while retaining data utility for analysis or development. Techniques include
encryption, substitution, and scrambling, ensuring compliance and minimizing breach risks.

## **Module Parameters**

- **Not Applicable (N/A):** There are no direct module parameters specified for the `ner_pii_masking` module.

## **Example config.yaml**

```yaml
modules:
  - module_type: ner_pii_masking
```
