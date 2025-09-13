# SNIFF VALIDATION ENGINE REPORT (v4)

## Protocol (narrative)
Children/adolescents undergoing minimally invasive repair of pectus excavatum (Nuss/MIRPE) receiving
intercostal nerve cryoablation (INC) intraoperatively for analgesia, compared to thoracic epidural,
paravertebral block, intercostal nerve block, erector spinae plane block, or systemic multimodal
analgesia, for postoperative opioid consumption and pain scores.

**Deterministic filters**: languages=english, spanish, portuguese, french, german, italian, chinese, japanese, korean ; year_min=2000

**Universe baseline query**:

```
("children adolescents minimally invasive"[tiab] OR nuss/mirpe[tiab]) AND ("intercostal nerve cryoablation analgesia"[tiab] OR inc[tiab]) AND ("children/adolescents undergoing minimally invasive"[tiab] OR "intercostal nerve cryoablation (INC)"[tiab] OR "thoracic epidural"[tiab] OR "paravertebral block"[tiab] OR "intercostal nerve block"[tiab] OR "erector spinae plane block"[tiab] OR "systemic multimodal analgesia"[tiab] OR "postoperative opioid consumption (in-hospital"[tiab] OR "pain scores"[tiab])
```

**Recommended filters**:

- topic_filter: <none>
- design_filter: "Randomized Controlled Trial"[Publication Type]

**Ground truth (confirmed) includes** (n=1): 40818798

**MeSH vernacular (from confirmed)**:

```
{
  "P": [],
  "I": [],
  "C": [],
  "O": [],
  "G": []
}
```

**WARNINGS**
- Insufficient confirmed includes after plausibility (0<3).
