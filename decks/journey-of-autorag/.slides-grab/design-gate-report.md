# slides-grab Design Gate Report

Verdict: proceed
Generated: 2026-07-02T08:54:28.333Z
Slide mode: presentation
Resolution: 2160p

## Pass A: System Contract / Constraint Integrity

# Pass A: System Contract / Constraint Integrity

Reviewed all 21 slides against the slides-grab system contract at 720pt x 405pt. Playwright validate reports 0 critical errors across all slides; every element stays inside the frame and no text is clipped. All imagery is local under ./assets/ with no remote image URLs.

- [x] System consistency: PASS — Shared dark token palette, Pretendard + Space Mono, and 720x405 frame applied uniformly to every slide.
- [x] Color discipline: PASS — Fixed accent set (green/cyan/amber/red) used semantically; no ad-hoc colors introduced.
- [x] AI slop tropes: PASS — No generic gradients-on-everything, no filler stock imagery; all figures are real repo/benchmark assets.
- [x] Content discipline: PASS — Claims map to source READMEs, star-history, NIPA proposal, and the HippoCamp benchmark numbers.

Confidence: High
Evidence: gate-preview/slide-01.png through gate-preview/slide-21.png (21 rendered 2160p PNGs).
Unresolved Critical: 0
Blocking findings: None

## Rendered evidence + slide fingerprints

- slide-01.html: f9a0e13ad6973e9196760b1f31140f6415c698d8d9edec29583b83c8eb2e76e6 — evidence gate-preview/slide-01.png
- slide-02.html: 0106c80ff364e46f8a1cc9fae6e714eb838ba7a1e09dfc1baeecec5487e5de71 — evidence gate-preview/slide-02.png
- slide-03.html: 887adcd81a9d74871274dae483e7eb16227023c25909b74779323df926ac0191 — evidence gate-preview/slide-03.png
- slide-04.html: 80f8d614c3196f9e391f351199e1be0cbdb97629c8413333c723b989cea32cbe — evidence gate-preview/slide-04.png
- slide-05.html: bb5d8c82cef8fffd2e946071bc8a5e4c6674b9825fc9c014c6012ae185123dad — evidence gate-preview/slide-05.png
- slide-06.html: 70e90410c532f30729ee8612954b34de807980bc11d20a10df6fb661ea02e0f1 — evidence gate-preview/slide-06.png
- slide-07.html: 6a4a456fade7f037e2d1c7aeb31c0b320c70e3777f109563c0442990bc4cd547 — evidence gate-preview/slide-07.png
- slide-08.html: 73feaa6e7697a5bd7ee66a67bb5016b45ec243a4da9b67dde4ea313b9e4de243 — evidence gate-preview/slide-08.png
- slide-09.html: 19312e1fc39a3dcfbfead0d40699196addee76ca8959b888d5a76ce5a7e8bad5 — evidence gate-preview/slide-09.png
- slide-10.html: 99b3d55efa5c49f792419d38f51c108b8bbb62d709a1edad66480e09842d1ba5 — evidence gate-preview/slide-10.png
- slide-11.html: ce35752addbac50137d865aad669ea5be09a34eb4249d053e09a2bb0ff6802dd — evidence gate-preview/slide-11.png
- slide-12.html: 0302907edc7e6c7fb66363b626586bf2f2d69981bc92c30941d423db6ff55a00 — evidence gate-preview/slide-12.png
- slide-13.html: 6b1442a8535cdf6f589ef7f113b24371ea83588a08d34ee8ac562282732bc8ab — evidence gate-preview/slide-13.png
- slide-14.html: f53aa13d6d8e2b77ee8df9d69f574c475f5f1504fc4f64a58d094799707f330b — evidence gate-preview/slide-14.png
- slide-15.html: 6fc77510a1b69de0588516ed492837fe1760d4139b9bbdaa3f328ab87c2b900f — evidence gate-preview/slide-15.png
- slide-16.html: 4d2d4f1ec2d97668618e9cac9be23b7cb6ae5277cdd1ef603651a0ce1b419389 — evidence gate-preview/slide-16.png
- slide-17.html: f9565c61f490c8ae25fbddcc450274d7206bbc25010e64e768055c45890870b6 — evidence gate-preview/slide-17.png
- slide-18.html: 2c6ad495e8d7cab3e5819b5c2d79b5bdb179443b1eafbcb220bcf89f96edb590 — evidence gate-preview/slide-18.png
- slide-19.html: 6613ad9e0594408d40e8414856f41eb5eb822340c449c0b948ea5ba01439b132 — evidence gate-preview/slide-19.png
- slide-20.html: 4b76742b5d5749f555b6254d99ee5afc4d3c217ce7e7af3bd72094dc4b62371a — evidence gate-preview/slide-20.png
- slide-21.html: 6dc5e4492897d5d478d14654854dbc9de2ba102564203feebace5e24792fd65f — evidence gate-preview/slide-21.png

## Findings

| Slide | Finding | Severity | Fix | Status |
| --- | --- | --- | --- | --- |
| slide-03 | Star-history chart sits with generous whitespace inside its white card. | Note | Acceptable — keeps the curve legible; left as-is. | Accepted |
| slide-21 | Two cosmetic sibling-overlap warnings on centered inline tag row. | Minor | Non-blocking; flex+gap layout keeps render clean. | Accepted |

VERDICT: PASS

## Pass B: Audience Impact / Expressive Readability

# Pass B: Audience Impact / Expressive Readability

Reviewed all 21 slides for a live 20-minute English talk to an AI-search/retrieval audience. Hierarchy, typography, and pacing read clearly at projection scale; the three-act journey (AutoML -> research automation -> search infra for agents) is easy to follow.

- [x] Composition & hierarchy: PASS — Kicker -> headline -> support -> evidence rhythm is consistent; one idea per slide.
- [x] Typography & legibility: PASS — Heading/body sizes verified against 4K renders; no clipped or cramped text.
- [x] Korean/CJK word-break integrity: PASS — Deck is all-English; no CJK line-break hazards present.
- [x] Review Litmus: PASS — Every slide earns its place in the 20-minute arc and advances the narrative.

Confidence: High
Evidence: gate-preview/slide-01.png through gate-preview/slide-21.png (21 rendered 2160p PNGs).
Unresolved Critical: 0
Blocking findings: None

## Rendered evidence + slide fingerprints

- slide-01.html: f9a0e13ad6973e9196760b1f31140f6415c698d8d9edec29583b83c8eb2e76e6 — evidence gate-preview/slide-01.png
- slide-02.html: 0106c80ff364e46f8a1cc9fae6e714eb838ba7a1e09dfc1baeecec5487e5de71 — evidence gate-preview/slide-02.png
- slide-03.html: 887adcd81a9d74871274dae483e7eb16227023c25909b74779323df926ac0191 — evidence gate-preview/slide-03.png
- slide-04.html: 80f8d614c3196f9e391f351199e1be0cbdb97629c8413333c723b989cea32cbe — evidence gate-preview/slide-04.png
- slide-05.html: bb5d8c82cef8fffd2e946071bc8a5e4c6674b9825fc9c014c6012ae185123dad — evidence gate-preview/slide-05.png
- slide-06.html: 70e90410c532f30729ee8612954b34de807980bc11d20a10df6fb661ea02e0f1 — evidence gate-preview/slide-06.png
- slide-07.html: 6a4a456fade7f037e2d1c7aeb31c0b320c70e3777f109563c0442990bc4cd547 — evidence gate-preview/slide-07.png
- slide-08.html: 73feaa6e7697a5bd7ee66a67bb5016b45ec243a4da9b67dde4ea313b9e4de243 — evidence gate-preview/slide-08.png
- slide-09.html: 19312e1fc39a3dcfbfead0d40699196addee76ca8959b888d5a76ce5a7e8bad5 — evidence gate-preview/slide-09.png
- slide-10.html: 99b3d55efa5c49f792419d38f51c108b8bbb62d709a1edad66480e09842d1ba5 — evidence gate-preview/slide-10.png
- slide-11.html: ce35752addbac50137d865aad669ea5be09a34eb4249d053e09a2bb0ff6802dd — evidence gate-preview/slide-11.png
- slide-12.html: 0302907edc7e6c7fb66363b626586bf2f2d69981bc92c30941d423db6ff55a00 — evidence gate-preview/slide-12.png
- slide-13.html: 6b1442a8535cdf6f589ef7f113b24371ea83588a08d34ee8ac562282732bc8ab — evidence gate-preview/slide-13.png
- slide-14.html: f53aa13d6d8e2b77ee8df9d69f574c475f5f1504fc4f64a58d094799707f330b — evidence gate-preview/slide-14.png
- slide-15.html: 6fc77510a1b69de0588516ed492837fe1760d4139b9bbdaa3f328ab87c2b900f — evidence gate-preview/slide-15.png
- slide-16.html: 4d2d4f1ec2d97668618e9cac9be23b7cb6ae5277cdd1ef603651a0ce1b419389 — evidence gate-preview/slide-16.png
- slide-17.html: f9565c61f490c8ae25fbddcc450274d7206bbc25010e64e768055c45890870b6 — evidence gate-preview/slide-17.png
- slide-18.html: 2c6ad495e8d7cab3e5819b5c2d79b5bdb179443b1eafbcb220bcf89f96edb590 — evidence gate-preview/slide-18.png
- slide-19.html: 6613ad9e0594408d40e8414856f41eb5eb822340c449c0b948ea5ba01439b132 — evidence gate-preview/slide-19.png
- slide-20.html: 4b76742b5d5749f555b6254d99ee5afc4d3c217ce7e7af3bd72094dc4b62371a — evidence gate-preview/slide-20.png
- slide-21.html: 6dc5e4492897d5d478d14654854dbc9de2ba102564203feebace5e24792fd65f — evidence gate-preview/slide-21.png

## Findings

| Slide | Finding | Severity | Fix | Status |
| --- | --- | --- | --- | --- |
| slide-08 | Nodes-and-modules figure is dense at a glance. | Note | Presenter narrates it; supporting caption added. | Accepted |
| slide-20 | Benchmark bars use a capped 76% max width for the raw baseline. | Note | Intentional so short jikji bars and value labels stay readable. | Accepted |

VERDICT: PASS

## Slide Fingerprints

- slide-01.html: f9a0e13ad6973e9196760b1f31140f6415c698d8d9edec29583b83c8eb2e76e6
- slide-02.html: 0106c80ff364e46f8a1cc9fae6e714eb838ba7a1e09dfc1baeecec5487e5de71
- slide-03.html: 887adcd81a9d74871274dae483e7eb16227023c25909b74779323df926ac0191
- slide-04.html: 80f8d614c3196f9e391f351199e1be0cbdb97629c8413333c723b989cea32cbe
- slide-05.html: bb5d8c82cef8fffd2e946071bc8a5e4c6674b9825fc9c014c6012ae185123dad
- slide-06.html: 70e90410c532f30729ee8612954b34de807980bc11d20a10df6fb661ea02e0f1
- slide-07.html: 6a4a456fade7f037e2d1c7aeb31c0b320c70e3777f109563c0442990bc4cd547
- slide-08.html: 73feaa6e7697a5bd7ee66a67bb5016b45ec243a4da9b67dde4ea313b9e4de243
- slide-09.html: 19312e1fc39a3dcfbfead0d40699196addee76ca8959b888d5a76ce5a7e8bad5
- slide-10.html: 99b3d55efa5c49f792419d38f51c108b8bbb62d709a1edad66480e09842d1ba5
- slide-11.html: ce35752addbac50137d865aad669ea5be09a34eb4249d053e09a2bb0ff6802dd
- slide-12.html: 0302907edc7e6c7fb66363b626586bf2f2d69981bc92c30941d423db6ff55a00
- slide-13.html: 6b1442a8535cdf6f589ef7f113b24371ea83588a08d34ee8ac562282732bc8ab
- slide-14.html: f53aa13d6d8e2b77ee8df9d69f574c475f5f1504fc4f64a58d094799707f330b
- slide-15.html: 6fc77510a1b69de0588516ed492837fe1760d4139b9bbdaa3f328ab87c2b900f
- slide-16.html: 4d2d4f1ec2d97668618e9cac9be23b7cb6ae5277cdd1ef603651a0ce1b419389
- slide-17.html: f9565c61f490c8ae25fbddcc450274d7206bbc25010e64e768055c45890870b6
- slide-18.html: 2c6ad495e8d7cab3e5819b5c2d79b5bdb179443b1eafbcb220bcf89f96edb590
- slide-19.html: 6613ad9e0594408d40e8414856f41eb5eb822340c449c0b948ea5ba01439b132
- slide-20.html: 4b76742b5d5749f555b6254d99ee5afc4d3c217ce7e7af3bd72094dc4b62371a
- slide-21.html: 6dc5e4492897d5d478d14654854dbc9de2ba102564203feebace5e24792fd65f
