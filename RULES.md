# 竞赛规则 / Competition Rules

> **来源说明**：本文件内容由参赛者从 Kaggle 竞赛页面
> （3LC Multi Vehicle detection Challenge → Rules 页）手动复制粘贴而来，
> 保留中英对照原文。**除文末「附录」外，均为规则原文，未经改写。**
>
> - 官方页面：https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/rules
> - 整理时间：2026-10-15
> - 若本文件与官方页面不一致，**以官方页面为准**。

---

ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.

参加本次竞赛即构成您接受这些官方竞赛规则。

Eligibility, prizes, and general legal terms follow standard Kaggle and sponsor policies. Below is the competition-specific summary.

资格、奖品和一般法律条款遵循标准的 Kaggle 和赞助商政策。以下是竞赛特定摘要。

## Competition-specific rules / 竞赛特定规则

### Task 任务

- **Multi-class object detection** with **four** vehicle classes (see Description / `dataset.yaml`).

  包含四个车辆类别的多类别目标检测（见描述 / `dataset.yaml`）。

- **Submissions**: CSV with columns `id`, `image_id`, `prediction_string` format in Evaluation.

  提交内容：CSV 文件，包含 `id`、`image_id`、`prediction_string` 列，格式见评估部分。

### Model and training constraints 模型与训练约束

- **Architecture**: **YOLOv8n only**. The official `train.py` is locked to YOLOv8n and trains **from scratch** (`yolov8n.yaml`, random initialization — no COCO or other pretrained checkpoints). Using larger YOLO variants, other architectures, or **replacing the training entrypoint with an unapproved script is not allowed**.

  架构：仅限 YOLOv8n。官方 `train.py` 锁定为 YOLOv8n，并从零开始训练（`yolov8n.yaml`，随机初始化——不使用 COCO 或其他预训练检查点）。不允许使用更大的 YOLO 变体、其他架构，或用未经批准的脚本替换训练入口点。

- **Input size**: **640 px** (default in `config.yaml`).

  输入尺寸：640 px（`config.yaml` 中的默认值）。

- **Hyperparameters**: **May be tuned within `config.yaml`** (learning rate, augmentation, epochs, etc.).

  超参数：可在 `config.yaml` 内进行调整（学习率、数据增强、轮数等）。

- **Prohibited techniques**: **Model ensembling, test-time augmentation (TTA), pseudo-labeling, knowledge distillation**, or any method that effectively circumvents the single-model constraint.

  禁止使用的技术：模型集成、测试时增强（TTA）、伪标签、知识蒸馏，或任何实际上规避单模型约束的方法。

### Data use 数据使用

- **Train / val**: Use only the provided data for training and local validation.

  训练/验证：仅使用提供的数据进行训练和本地验证。

- **Test**: Images only — do not attempt to obtain or use leaked labels.

  测试：仅使用图像，不要尝试获取或使用泄露的标签。

- **External data / images**: **Not allowed** unless an explicit exception is published by the organizers.

  外部数据/图像：除非主办方发布明确的例外，否则不允许使用。

- **Pretrained weights**: **Not allowed**. Training must start from random initialization.

  预训练权重：不允许。训练必须从随机初始化开始。

### 3LC

- **3LC is required for the prescribed workflow**: register train and val Tables, run training integrated with 3LC Runs, and use the Dashboard for the data-centric loop.

  规定的工作流程要求使用 3LC：注册 train 和 val Tables，运行与 3LC Runs 集成的训练，并使用 Dashboard 进行以数据为中心的循环。

- Test images are **not** registered in 3LC; inference uses `predict.py` on local test files.

  测试图像未在 3LC 中注册；推理使用 `predict.py` 在本地测试文件上进行。

- Participants must use their **own 3LC account and API key**.

  参赛者必须使用自己的 3LC 账户和 API 密钥。

### Teams and submissions 团队与提交

- **Team size**: up to 4 participants.

  团队规模：最多 4 名参与者。

- **Submissions**: **3 per day maximum**.

  提交次数：每天最多 3 次。

- **Final selection**: Choose up to **2 submissions** for final judging (per Kaggle settings).

  最终选择：最多选择 2 份提交用于最终评审（根据 Kaggle 设置）。

### Timeline 时间安排

- **Start / end / merger deadline**: As shown on the competition Timeline.

  开始/结束/合并截止时间：见竞赛 Timeline 页面。

### Evaluation form (if applicable)

- Teams are required to submit an **evaluation / feedback form** and a **private GitHub repo** by the stated deadline to be eligible for **prizes and certificates**.

  参赛队伍须在截止日期前提交评估/反馈表和**私有 GitHub 仓库**，才有资格获得奖品和证书。

- The form link, judge GitHub Username, and repo-contents checklist provided in Overview section.

  表格链接、评委 GitHub 用户名和仓库内容清单在 Overview 部分提供。

- **Evaluation Google Form URL**: （原文此处仅给出字段名，未附链接）

### Winner obligations

- Provide **reproducible code and methodology** as required by the sponsor.

  按赞助商要求提供**可复现的代码和方法论**。

- Grant license to the winning submission as stated in the full legal Rules.

  按完整法律规则授予获胜提交的许可。

> 一般 Kaggle 竞赛规则和法律条款请参考 Kaggle Competition Rules。
> 原文此处有笔误："Overview sectom" 应为 "Overview section"。

---

## Kaggle Competition Foundational Rules（不可编辑）

> 原文说明：Competition participants must also agree to Kaggle's Foundational Competition Rules. These rules will **supersede** the competition-specific rules in the event of any conflict.
>
> 参赛者还须同意 Kaggle 基础竞赛规则。**如有冲突，以基础规则为准。**

The following Kaggle Competition Foundational Rules ("Foundational Rules") apply to every competition regardless of whether the Sponsor creates competition-specific rules. Any competition-specific rules provided by the Sponsor are in addition to these rules, and in the case of any conflict or inconsistency, these Foundational Rules control and nullify contrary competition-specific rules.

以下 Kaggle 基础竞赛规则适用于所有竞赛。赞助商制定的竞赛特定规则是对本规则的补充；**如有冲突或不一致，以本基础规则为准，并使其相冲突的特定规则失效。**

### GENERAL COMPETITION RULES - BINDING AGREEMENT

**1. ELIGIBILITY**

a. To be eligible to enter the Competition, you must be:
- a registered account holder at Kaggle.com;
- the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
- not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, or North Korea; and
- not a person or representative of an entity under U.S. export controls or sanctions (see: https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx).

b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

**2. SPONSOR AND HOSTING PLATFORM**

a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at www.kaggle.com/terms in addition to these Rules.

**3. COMPETITION PERIOD**

a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the "Competition Period"). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

**4. COMPETITION ENTRY**

a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize.

b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.

c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible.

d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

**5. INDIVIDUALS AND TEAMS**

a. **Individual Account.** You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website.

b. **Teams.** If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website.

c. **Team Merger.** Teams may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules.

d. **Private Sharing.** No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

**6. SUBMISSION CODE REQUIREMENTS**

a. **Private Code Sharing.** Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition ("Competition Code"). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification.

b. **Public Code Sharing.** You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code.

c. **Use of Open Source.** Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

**7. DETERMINING WINNERS**

a. Each Submission will be scored and ranked by the evaluation metric stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set.

b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.

**8. NOTIFICATION OF WINNERS & DISQUALIFICATION**

a. The potential winner(s) will be notified by email.

b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition's judging criteria.

c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant's compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph.

d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle.

e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded.

f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

**9. PRIZES**

a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants.

b. All Prizes are subject to Competition Sponsor's review and verification of the Participant's eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions).

c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8.

d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed.

e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above.

f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

**10. TAXES**

a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

**11. GENERAL CONDITIONS**

a. All federal, state, provincial and local laws and regulations apply.

**12. PUBLICITY**

a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

**13. PRIVACY**

a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address ("Personal Information"). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>.

b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to the Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

**14. WARRANTY, INDEMNITY AND RELEASE**

a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law.

b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity.

c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

**15. INTERNET**

a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant's ability to participate in the Competition.

**16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY**

a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

**17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT**

a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

**18. DEFINITIONS**

a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants.

b. An "Entry" is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition.

c. A "Final Submission" is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard.

d. A "Participant" or "Participant User" is an individual who participates in a competition by entering the competition and making a Submission.

e. The "Private Leaderboard" is a ranked display of Participants' Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition.

f. The "Public Leaderboard" is a ranked display of Participants' Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition.

g. A "Sponsor" is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules.

h. A "Submission" is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor.

i. A "Team" is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.

---

---

## 附录：本项目操作要点（**非规则原文，由 AI 提炼，供施工参考**）

> 以下内容不是竞赛规则，是为了施工方便从上述规则中提炼的要点。
> **机器可执行的版本在 `AGENTS.md` §5「AI 操作授权清单」**，两者配合使用。
> 如有疑义，以本文件上方的规则原文为准。

### A. 硬性约束（违反即违规）

| # | 约束 | 规则依据 |
| :--- | :--- | :--- |
| 1 | 架构**只能用 YOLOv8n** | Model and training constraints |
| 2 | **必须从随机初始化训练**，禁用任何预训练权重（含 COCO） | Model and training constraints / Data use |
| 3 | **禁止替换官方 `train.py` 训练入口** | Model and training constraints |
| 4 | 输入尺寸锁定 **640 px** | Model and training constraints |
| 5 | **必须使用 3LC 工作流**：注册 train/val Tables、训练集成 3LC Runs、用 Dashboard 做数据循环 | 3LC |
| 6 | 必须使用**自己的 3LC 账号与 API key** | 3LC |
| 7 | 禁用：**模型集成、TTA、伪标签、知识蒸馏**，及一切规避单模型约束的方法 | Prohibited techniques |
| 8 | 训练/验证**只能用提供的数据**；禁用外部数据/图像 | Data use |
| 9 | 测试集**只有图像**，禁止获取或使用泄露标签 | Data use |
| 10 | 禁止对验证集或测试集做人工标注/人工预测 | Foundational Rules 4.b |
| 11 | 提交格式必须是 `id, image_id, prediction_string` 的 CSV | Task |

### B. 允许的操作空间

| # | 允许 | 规则依据 |
| :--- | :--- | :--- |
| 1 | 在 **`config.yaml` 内调参**：学习率、数据增强、epochs 等 | Hyperparameters |
| 2 | 在官方流程内**改进数据**（清洗标签、修正漏标、剔除坏样本），这正是 Dashboard 数据循环的用途 | 3LC |
| 3 | 编写自己的**分析/评估脚本**（不影响训练入口点的前提下） | 规则未限制 |
| 4 | 复用官方提供的 `verify_setup.py` / `register_tables.py` / `predict.py` | 官方提供 |

### C. 提交与评奖相关

| # | 事项 |
| :--- | :--- |
| 1 | 每天最多 **3 次**提交 |
| 2 | 最终最多选 **2 份**提交进入最终评审 |
| 3 | 评奖需在截止前提交 **evaluation/feedback form + 私有 GitHub 仓库** |
| 4 | 获奖需提供**可复现的代码与方法论** |
| 5 | 团队上限 4 人；每人只能加入 1 个团队 |

### D. 本项目对应的文件位置

- starter kit 根目录：`multi-vehicle_kaggle_competition_starter_kit_files/`
- 配置文件（调参主战场）：`multi-vehicle_kaggle_competition_starter_kit_files/config.yaml`
- 数据描述（类别/路径）：`multi-vehicle_kaggle_competition_starter_kit_files/dataset.yaml`
- 官方流程文档：`multi-vehicle_kaggle_competition_starter_kit_files/README.md`
