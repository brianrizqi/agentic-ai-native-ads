# Results summary

| run | n | acc [95% CI] | macro-F1 [95% CI] | MCC | invalid |
|---|---|---|---|---|---|
| gemma3-12b__reasoning_first/test__rag_k5_maxsim0.90 | 1000 | 0.9780 [0.9669, 0.9854] | 0.9780 [0.9680, 0.9870] | 0.9560 | 0 |
| gemma3-12b__reasoning_first/test__rag_k5_random | 1000 | 0.9760 [0.9645, 0.9838] | 0.9760 [0.9660, 0.9850] | 0.9521 | 0 |
| gemma3-12b__reasoning_first/test__rag_k5_minilm | 1000 | 0.9760 [0.9645, 0.9838] | 0.9760 [0.9660, 0.9850] | 0.9520 | 0 |
| gemma3-12b__reasoning_first/test__rag_k10 | 1000 | 0.9750 [0.9634, 0.9830] | 0.9750 [0.9640, 0.9840] | 0.9500 | 0 |
| gemma3-12b__reasoning_first/test__rag_k3 | 1000 | 0.9750 [0.9634, 0.9830] | 0.9750 [0.9650, 0.9840] | 0.9500 | 0 |
| gemma3-12b__reasoning_first/test__rag_k5_lang-cross | 1000 | 0.9740 [0.9622, 0.9822] | 0.9740 [0.9638, 0.9830] | 0.9480 | 0 |
| gemma3-12b__reasoning_first/test__rag_k5_intervene-flip_each | 500 | 0.9740 [0.9560, 0.9847] | 0.9739 [0.9597, 0.9860] | 0.9478 | 0 |
| gemma3-12b__reasoning_first/test__rag_k5_nolabels | 1000 | 0.9730 [0.9610, 0.9814] | 0.9730 [0.9628, 0.9829] | 0.9459 | 0 |
| gemma3-12b__reasoning_first/test__rag_k1 | 1000 | 0.9710 [0.9587, 0.9797] | 0.9710 [0.9598, 0.9810] | 0.9419 | 0 |
| gemma3-12b__reasoning_first/test__rag_k5_notext | 1000 | 0.9710 [0.9587, 0.9797] | 0.9710 [0.9600, 0.9810] | 0.9419 | 0 |
| gemma3-12b__reasoning_first/test__rag_k5 | 2165 | 0.9709 [0.9629, 0.9772] | 0.9709 [0.9635, 0.9778] | 0.9420 | 0 |
| gemma3-4b__reasoning_first/test__rag_k10 | 1000 | 0.9700 [0.9575, 0.9789] | 0.9700 [0.9588, 0.9800] | 0.9401 | 0 |
| gemma3-12b__reasoning_first/test__norag | 2165 | 0.9686 [0.9604, 0.9751] | 0.9686 [0.9612, 0.9759] | 0.9373 | 0 |
| qwen3.5-9b__reasoning_first/test__rag_k5 | 2165 | 0.9681 [0.9599, 0.9747] | 0.9681 [0.9603, 0.9751] | 0.9364 | 0 |
| gemma2-9b__reasoning_first/test__rag_k5 | 2165 | 0.9663 [0.9578, 0.9731] | 0.9663 [0.9584, 0.9736] | 0.9327 | 0 |
| gemma3-4b__reasoning_first/test__rag_k3 | 1000 | 0.9650 [0.9517, 0.9747] | 0.9650 [0.9530, 0.9760] | 0.9300 | 0 |
| qwen3-8b__reasoning_first/test__rag_k5 | 2165 | 0.9649 [0.9563, 0.9719] | 0.9649 [0.9570, 0.9727] | 0.9301 | 0 |
| qwen3.5-9b__reasoning_first/test__norag | 2165 | 0.9630 [0.9542, 0.9702] | 0.9630 [0.9547, 0.9708] | 0.9268 | 0 |
| qwen2.5-14b__reasoning_first/test__rag_k5 | 2165 | 0.9626 [0.9537, 0.9698] | 0.9626 [0.9543, 0.9700] | 0.9259 | 0 |
| gemma2-9b__reasoning_first/test__norag | 2165 | 0.9617 [0.9527, 0.9690] | 0.9617 [0.9538, 0.9695] | 0.9234 | 0 |
| qwen3.5-2b__reasoning_first/test__rag_k5 | 2165 | 0.9612 [0.9522, 0.9686] | 0.9612 [0.9528, 0.9691] | 0.9228 | 0 |
| gemma3-4b__reasoning_first/test__rag_k1 | 1000 | 0.9600 [0.9460, 0.9705] | 0.9599 [0.9479, 0.9710] | 0.9199 | 0 |
| qwen3-8b__reasoning_first/test__norag | 2165 | 0.9594 [0.9502, 0.9669] | 0.9594 [0.9510, 0.9672] | 0.9188 | 0 |
| gemma3-4b__reasoning_first/test__rag_k5 | 2165 | 0.9589 [0.9497, 0.9665] | 0.9589 [0.9505, 0.9667] | 0.9180 | 0 |
| qwen2.5-14b__reasoning_first/test__norag | 2165 | 0.9566 [0.9472, 0.9644] | 0.9566 [0.9481, 0.9649] | 0.9136 | 0 |
| gemma3-4b__reasoning_first/test__norag | 2165 | 0.9520 [0.9421, 0.9602] | 0.9520 [0.9423, 0.9612] | 0.9039 | 0 |
| llama3.2-1b__reasoning_first/test__rag_k5 | 2165 | 0.9515 [0.9416, 0.9598] | 0.9515 [0.9418, 0.9602] | 0.9030 | 0 |
| llama3.2-1b__reasoning_first/test__norag | 2165 | 0.9469 [0.9366, 0.9556] | 0.9469 [0.9376, 0.9561] | 0.8939 | 0 |
| gemma3-12b__reasoning_first/test__rag_k5_flipped | 1000 | 0.9450 [0.9291, 0.9575] | 0.9449 [0.9307, 0.9590] | 0.8900 | 0 |
| qwen3.5-2b__reasoning_first/test__norag | 2165 | 0.9436 [0.9331, 0.9526] | 0.9436 [0.9335, 0.9533] | 0.8874 | 0 |
| gemma3-1b__reasoning_first/test__rag_k5 | 2165 | 0.9436 [0.9331, 0.9526] | 0.9436 [0.9339, 0.9533] | 0.8877 | 0 |
| deepseek-r1-llama-8b__reasoning_first/test__rag_k5 | 2165 | 0.9413 [0.9306, 0.9505] | 0.9413 [0.9306, 0.9510] | 0.8828 | 0 |
| gemma3-270m__reasoning_first/test__norag | 2165 | 0.9349 [0.9237, 0.9445] | 0.9381 [0.9279, 0.9480] | 0.8829 | 15 |
| gemma3-270m__reasoning_first/test__rag_k5 | 2165 | 0.9316 [0.9202, 0.9415] | 0.9359 [0.9256, 0.9460] | 0.8810 | 20 |
| deepseek-r1-llama-8b__reasoning_first/test__norag | 2165 | 0.9335 [0.9222, 0.9432] | 0.9335 [0.9224, 0.9436] | 0.8672 | 0 |
| gemma3-1b__reasoning_first/test__norag | 2165 | 0.9321 [0.9207, 0.9420] | 0.9320 [0.9213, 0.9422] | 0.8662 | 0 |

| A | B | b | c | diff [95% CI] | p exact | p Holm |
|---|---|---|---|---|---|---|
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k5_random | 6 | 4 | -0.0020 [-0.0080, +0.0040] | 0.754 | 1 |
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k5_flipped | 37 | 4 | -0.0330 [-0.0460, -0.0210] | 1.03e-07 | 1.33e-06 |
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k5_nolabels | 6 | 1 | -0.0050 [-0.0100, +0.0000] | 0.125 | 1 |
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k5_notext | 8 | 1 | -0.0070 [-0.0130, -0.0020] | 0.0391 | 0.469 |
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k5_maxsim0.90 | 0 | 0 | +0.0000 [+0.0000, +0.0000] | 1 | 1 |
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k5_lang-cross | 7 | 3 | -0.0040 [-0.0100, +0.0020] | 0.344 | 1 |
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k5_minilm | 3 | 1 | -0.0020 [-0.0060, +0.0010] | 0.625 | 1 |
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k1 | 8 | 1 | -0.0070 [-0.0130, -0.0020] | 0.0391 | 0.469 |
| gemma3-4b__reasoning_first/test__rag_k5 | gemma3-4b__reasoning_first/test__rag_k1 | 12 | 7 | -0.0050 [-0.0130, +0.0030] | 0.359 | 1 |
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k3 | 4 | 1 | -0.0030 [-0.0080, +0.0010] | 0.375 | 1 |
| gemma3-4b__reasoning_first/test__rag_k5 | gemma3-4b__reasoning_first/test__rag_k3 | 4 | 4 | +0.0000 [-0.0060, +0.0060] | 1 | 1 |
| gemma3-12b__reasoning_first/test__rag_k5 | gemma3-12b__reasoning_first/test__rag_k10 | 4 | 1 | -0.0030 [-0.0080, +0.0010] | 0.375 | 1 |
| gemma3-4b__reasoning_first/test__rag_k5 | gemma3-4b__reasoning_first/test__rag_k10 | 3 | 8 | +0.0050 [-0.0010, +0.0120] | 0.227 | 1 |
