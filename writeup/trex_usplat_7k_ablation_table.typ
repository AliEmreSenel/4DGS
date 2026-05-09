#figure(
  block(width: 100%)[
    #set text(size: 5.5pt)

    #let dark = rgb("#2c3e50")
    #let section = rgb("#f2f2f2")
    #let border = rgb("#cccccc")

    #table(
      columns: (0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr, 0.65fr, 0.65fr, 0.65fr, 0.65fr, 0.65fr, 0.65fr, 0.65fr, 0.65fr, 0.65fr),
      stroke: border,
      inset: 3pt,
      align: center + horizon,

      table.cell(colspan: 5, fill: section)[#text(weight: "bold")[ABLATIONS]],
      table.cell(colspan: 9, fill: section)[#text(weight: "bold")[VISUAL]],

      table.cell(fill: dark)[#text(fill: white, weight: "bold")[Iso]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[App]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[Prune]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[Drop]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[ESS]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[Iter]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[PSNR ↑]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[SSIM ↑]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[LPIPS ↓]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[FPS ↑]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[VRAM ↓]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[MB ↓]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[Gauss k ↓]],
      table.cell(fill: dark)[#text(fill: white, weight: "bold")[Train ↓]],

      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [29.41], [0.976], [0.038], [333], [414], [274.4], [157.4], [490s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [29.40], [0.977], [0.032], [318], [447], [297.0], [170.3], [295s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [30.24], [0.981], [0.025], [315], [467], [316.5], [170.3], [534s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [29.52], [0.979], [0.028], [372], [440], [297.6], [160.1], [286s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [28.98], [0.976], [0.031], [358], [476], [324.9], [174.8], [299s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [29.51], [0.979], [0.029], [360], [430], [288.2], [155.1], [502s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [29.22], [0.978], [0.031], [334], [477], [325.6], [175.2], [525s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [30.01], [0.981], [0.023], [624], [92], [38.4], [156.8], [118s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [29.75], [0.980], [0.026], [587], [92], [38.2], [156.3], [263s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [29.72], [0.980], [0.026], [593], [95], [40.5], [165.4], [263s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [29.00], [0.978], [0.027], [733], [94], [39.2], [160.2], [114s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [28.92], [0.978], [0.029], [725], [98], [41.9], [171.3], [113s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [28.76], [0.978], [0.029], [645], [96], [40.7], [166.2], [255s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [28.30], [0.976], [0.032], [653], [101], [43.0], [175.7], [255s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [29.73], [0.981], [0.024], [634], [95], [40.7], [166.4], [120s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [29.34], [0.976], [0.034], [321], [448], [298.9], [171.4], [503s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [30.51], [0.982], [0.026], [329], [431], [291.3], [156.8], [516s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [29.19], [0.976], [0.034], [354], [421], [279.2], [160.1], [280s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [29.15], [0.975], [0.034], [339], [456], [305.2], [175.0], [296s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [29.34], [0.976], [0.033], [340], [431], [287.2], [164.7], [493s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [29.11], [0.975], [0.034], [339], [449], [299.3], [171.6], [495s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [28.67], [0.975], [0.035], [600], [74], [19.6], [150.0], [116s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [28.37], [0.974], [0.036], [580], [78], [22.1], [169.8], [114s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [30.63], [0.983], [0.023], [341], [448], [301.4], [162.2], [292s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [28.38], [0.973], [0.040], [595], [77], [21.2], [163.0], [241s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [28.49], [0.975], [0.033], [680], [76], [20.9], [160.2], [106s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [28.54], [0.975], [0.033], [679], [80], [22.6], [173.5], [108s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [28.24], [0.974], [0.034], [621], [78], [21.8], [167.6], [239s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#ffe0b2"))[prune+dense], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#d9efdf"))[yes], [20000], [28.28], [0.974], [0.035], [639], [80], [22.5], [173.0], [237s],
      table.cell(fill: rgb("#fff4bf"))[aniso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [30.54], [0.982], [0.023], [362], [415], [279.4], [150.3], [284s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#fce4ec"))[RGB], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#d9efdf"))[yes], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [28.44], [0.974], [0.038], [591], [75], [19.9], [152.5], [238s],
      table.cell(fill: rgb("#e3f2fd"))[iso], table.cell(fill: rgb("#ede7f6"))[SH3], table.cell(fill: rgb("#f2f2f2"))[none], table.cell(fill: rgb("#f2f2f2"))[no], table.cell(fill: rgb("#f2f2f2"))[no], [20000], [29.33], [0.977], [0.032], [330], [413], [273.9], [157.1], [282s],
    )
  ],
  caption: [trex ablation results.],
)