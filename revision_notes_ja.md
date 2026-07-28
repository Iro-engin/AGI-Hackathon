# ATLAS NeurIPS版への再構成メモ

## 1. 再構成の結論

ICML版の広い主張を維持したまま実験を追加する方針は採らなかった。元稿では、合成実験の情報時点、理論上の閾値と実装上の閾値、change-delay証明の中心、missing-data実験と提案GEM、時変pilotとオンライン計算量の間に不一致があった。NeurIPS版では、これらを表面的に修正せず、論文を独立したTheory contributionとして再定義した。

新しい中心命題は次の一つである。

> 固定pilotの直交残差空間で、重尾に頑健なmultiscale EWMAを用いて一期間先共分散を予測し、未知の局所Hölder smoothnessへ適応しながら、Gaussian rank-one部分モデル上のminimax rateを対数因子まで達成する。

## 2. 主な変更

### 厳密な情報時点

時点`t`の観測を更新へ使った後、`t+1`の条件付き共分散を予測する。same-time independent replicateを一期間先予測の証拠として扱わない。

### frozen-pilot episode

`B, D, Q`をdeployment episode内で固定する。pilotを再推定した場合、別episodeとしてscatterをリセットする。これにより、過去データを異なる座標系で混合する問題を除き、`O(Jm^2)`の逐次更新を定義通り実行できる。

### 理論とアルゴリズムの一致

Lepski ruleは本文と実装で同じ高確率閾値

`4 a_t(h) + 2 b_tau`

を使用する。予測用の経験的定数をrank certificateへ流用しない。予測出力と構造的なrank certificateを分離した。

### EWMA population targetの導入

推定誤差を、

1. sample scatterとEWMA population targetの差
2. clipping bias
3. EWMA targetと一期間先の瞬時共分散のlag/drift差

へ分解した。change-delayはEWMA population targetの周りで証明し、一期間先誤差にはlag項を別に加える。

### minimax lower bound

上界だけではNeurIPS Theory論文として弱いため、独立Gaussian streamかつrank-one residual covarianceという小さい部分モデル上でFano型lower boundを追加した。未知のsmoothnessに対する上界とdimension/smoothnessの次数が一致する。

### unsupported claimsの削除

次をNeurIPS版の主張から外した。

- full-covariance Student-t EMで代用したmissing-data優位性
- constrained pilot-preserving GEMの未検証主張
- contemporaneous held-out replicateに基づくone-step性能
- 一部のfixed-memory候補を除外した優位性
- 小さい実データ差に基づく一般的な性能優位性

本文は数値結果を捏造せず、実証優位性を主張しない。

## 3. NeurIPS版の主定理

1. pilot-relative identification
2. heavy-tail下のsimultaneous one-step operator bound
3. theorem-matched Lepski selectorのadaptive oracle inequality
4. unknown local Hölder smoothnessへの適応率
5. Gaussian rank-one部分モデル上のmatching minimax lower bound
6. honest rank/subspace certificate
7. negative-residual pilot-validity certificate
8. EWMA targetを基準にしたbirth/death delay
9. Gaussian quasi-likelihoodとquadratic decision regretへのtransfer bound

全証明は`proofs.tex`へ収録した。

## 4. NeurIPS向けの構成

本文は9ページ以内を目標とし、次の順序で一つのPDFへまとめる。

1. 本文
2. 参考文献
3. 技術Appendixと完全証明
4. Broader Impact
5. NeurIPS Paper Checklist

公式`neurips_2026.sty`を変更せず使用する。ビルドログには本文終了、Appendix開始、checklist開始ページを出力し、`page_audit.txt`で確認できる。

## 5. 残る査読リスク

### 新規性の境界

Lepski adaptation、robust covariance、online subspace trackingの各要素は既存である。採択には、それらの単純な組合せではなく、pilot-preserving one-step problem、lag-aware oracle inequality、rank-one lower bound、pilot-validity certificateが一体として新しいことをIntroductionとrelated workで明確にする必要がある。

### lower boundの独立検証

Fano packingの構成とHölder path scalingは整合するよう導出したが、投稿前に理論共著者が定数条件、packing reduction、pointwise prediction targetを独立に再検算すべきである。

### 実証を持たないTheory論文

NeurIPS Main TrackはTheoryを対象とするため実験は必須ではない。ただし、査読者がsignificanceを実証で判断する場合、評価が下がる可能性がある。本稿は不完全な実験を残すより、理論のsoundnessを優先した。

### one-sided residual

PSD correctionはpilotの過小評価だけを修正する。過大評価はnegative-residual flagで検出するが修正しない。signed correctionへ拡張すると識別とpositive definitenessに新しい条件が必要となる。

### complete observations

missing coordinatesは本稿の対象外である。将来拡張では、固定pilot mapを維持するmissing-data estimatorを導出し、そのアルゴリズムそのものを評価する必要がある。

## 6. 投稿前の必須確認

- `MAIN_END_PAGE <= 9`
- official 2026 style、US Letter、font embedding
- undefined citation/referenceがゼロ
- Appendixを含む全定理の独立検算
- anonymous metadataとsupplementary files
- 2026年版checklistの16項目

採択を保証することはできない。今回の再構成は、元稿でReject理由となるsoundness mismatchを除き、NeurIPS Theory査読で評価可能な一本の主張へ論文を集中させた。
