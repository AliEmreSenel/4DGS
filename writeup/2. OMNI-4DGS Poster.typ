#import "@preview/peace-of-posters:0.6.0" as pop
#import "@preview/fontawesome:0.6.0": *
#import "appendix.typ": contrib-table-large

#set page("a0", margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)

#set text(size: pop.layout-a0.at("body-size"))
#set par(justify: false, leading: 0.62em)

#let box-spacing = 1.2em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

// -----------------------------------------------------------------------------
// Geometry: after the title box, the content is split into:
// top area = 2/3, bottom area = 1/3.
// Tune these two values if your title/logo height changes.
// -----------------------------------------------------------------------------
#let top-h = 66.8cm
#let bottom-h = 34.4cm

// -----------------------------------------------------------------------------
// Colors
// -----------------------------------------------------------------------------
#let navy = rgb("#0b1026")
#let ink = rgb("#182033")
#let muted = rgb("#667085")
#let blue = rgb("#3466ff")
#let cyan = rgb("#18c7d9")
#let violet = rgb("#7c3aed")
#let green = rgb("#22a06b")
#let orange = rgb("#f97316")
#let red = rgb("#ef4444")
#let surface = rgb("#f7f9ff")
#let line-color = rgb("#d7def2")
#let bottom-blue = rgb("#dff0ff")

// -----------------------------------------------------------------------------
// Small helpers, styled to stay close to peace-of-posters blocks
// -----------------------------------------------------------------------------
#let tag(body, fill: surface, text-fill: blue) = box(
  fill: fill,
  stroke: 0.4pt + line-color,
  radius: 7pt,
  inset: (x: 7pt, y: 3.2pt),
)[
  #text(size: 1em, weight: "bold", fill: text-fill)[#body]
]

#let callout(title, body, fill: surface, accent: blue) = box(
  width: 100%,
  fill: fill,
  stroke: 1pt + accent,
  radius: 12pt,
  inset: 12pt,
)[
  #text(weight: "bold", fill: accent, size: 0.92em)[#title]
  #linebreak()
  #text(size: 0.78em)[#body]
]

#let mini-step(num, title, body, accent: blue) = grid(
  columns: (31pt, 1fr),
  gutter: 8pt,
  align: horizon,
  [
    #circle(fill: accent, inset: 5pt)[
      #text(fill: white, weight: "bold", size: 0.66em)[#num]
    ]
  ],
  [
    #text(weight: "bold", size: 0.82em)[#title]
    #linebreak()
    #text(size: 0.7em, fill: muted)[#body]
  ],
)

#let metric-pill(name, direction, fill: surface, text-fill: ink) = box(
  fill: fill,
  radius: 7pt,
  inset: (x: 7pt, y: 3pt),
)[#text(size: 0.8em, weight: "bold", fill: text-fill)[#name #direction]]

#let img-slot(title, subtitle, height: 10cm, path: none) = box(
  width: 100%,
  height: height,
  fill: gradient.linear(surface, white, angle: 25deg),
  stroke: (paint: blue, thickness: 1.1pt, dash: "dashed"),
  radius: 14pt,
  inset: 11pt,
)[
  #align(center + horizon)[
    #if path == none [
      #text(size: 0.95em, weight: "bold", fill: blue)[#title]
      #linebreak()
      #text(size: 0.68em, fill: muted)[#subtitle]

      #text(size: 0.6em, fill: muted)[Replace with `#image("img/...", width: 100%)`]
    ] else [
      #image(path, width: 100%, height: 100%, fit: "contain")
    ]
  ]
]

#let small-img(label, path: none) = box(
  width: 100%,
  height: 4.9cm,
  fill: white,
  stroke: (paint: line-color, thickness: 0.8pt, dash: "dashed"),
  radius: 10pt,
  inset: 6pt,
)[
  #align(center + horizon)[
    #if path == none [
      #text(size: 0.63em, fill: muted)[#label]
    ] else [
      #image(path, width: 100%, height: 100%, fit: "contain")
    ]
  ]
]

#let ref-line(key, entry) = [
  #text(size: 0.63em)[
    #text(weight: "bold")[#key] #entry
  ]
  #linebreak()
]

// Custom box styles derived from the theme.
#let result-heading-args = pop.uni-fr.heading-box-args
#result-heading-args.insert("fill", blue)

#let result-body-args = pop.uni-fr.body-box-args
#result-body-args.insert("fill", white)
#result-body-args.insert("inset", 18pt)

#let final-heading-args = pop.uni-fr.heading-box-args
#final-heading-args.insert("fill", navy)

#let final-body-args = pop.uni-fr.body-box-args
#final-body-args.insert("fill", white)
#final-body-args.insert("inset", 18pt)

// -----------------------------------------------------------------------------
// Title
// -----------------------------------------------------------------------------
#pop.title-box(
  [OMNI-4DGS: Fast, Light and Precise Video-to-Model Reconstruction],
  authors: "Ali Emre Senel¹ · Tebe Nigrelli¹ · Stefana Chiriac¹",
  institutes: "¹Bocconi University, Milan, Italy",
  keywords: "4D Gaussian Splatting · Dynamic Scene Reconstruction · Ablations · Efficient Rendering",
)



// -----------------------------------------------------------------------------
// TOP 2/3: three text columns
// -----------------------------------------------------------------------------
#box(width: 100%, height: top-h)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: box-spacing,

    // -------------------------------------------------------------------------
    // Column 1
    // -------------------------------------------------------------------------
    [
      #pop.column-box(heading: [Context #h(1fr) #fa-icon("map", solid: true)])[
        4D Gaussian Splatting (4DGS) is the dominant approach to dynamic scene reconstruction from video. #h(1fr) _Fast and flexible_ #h(1fr)
      ]

      #pop.column-box(heading: [Motivation #h(1fr) #fa-icon("clipboard-question")])[
        Existing methods usually optimize one axis:
        *fidelity*, *FPS*, *memory*, or *train time*.

        The best? _*We test the optimizations together*_


        #tag([Quality], fill: rgb("#edf5ff"), text-fill: blue)
        #h(0.35em)
        #tag([Speed], fill: rgb("#effcf6"), text-fill: green)
        #h(0.35em)
        #tag([Memory], fill: rgb("#fff7ed"), text-fill: orange)
        #h(0.35em)
        #tag([Stability], fill: rgb("#f5f3ff"), text-fill: violet)
      ]

      #pop.column-box(heading: [4D Gaussian Building Blocks #h(1fr) #fa-icon("puzzle-piece")])[
        Each primitive is centered in space-time and stores opacity, shape, rotation, and color. The native representation is expressive, but compact variants trade quality for speed and size.

        $
          G_i = (mu_i, Sigma_i, o_i, r_i, arrow("SH")_i),
          quad mu_i in RR^4,
          quad Sigma_i in RR^(4 times 4)
        $


        #table(
          columns: (0.8fr, 0.3fr, 2fr),
          inset: 10pt,
          align: (x, y) => if x == 1 { center } else { left },
          stroke: (x, y) => if y > 0 { (top: 0.35pt + line-color) },

          [*4D mean*], table.cell(fill: gray.lighten(85%))[$mu_i$], [mean is the gaussian center],
          [*Covariance*], table.cell(fill: gray.lighten(85%))[$Sigma_i$], [(sphere || ellipsoid) $=:$ *Isotropy*],
          [*Opacity*], table.cell(fill: gray.lighten(85%))[$o_i$], [adjusts visibility, see-through],
          [*Color*], table.cell(fill: gray.lighten(85%))[$"SH"_i$], [$"SH"(0) = "RGB" arrow.r.long.squiggly "SH"(3)$],
        )
      ]

      #pop.column-box(heading: [Training Objective #h(1fr) #fa-icon("route")])[
        By Ablation: fidelity, background, 4D motion.

        #strong[Photometric #h(1em)]
        reconstruct each train view:
        $
          cal(L)_"rgb" =
          underbrace((1 - lambda_"dssim") cal(L)_1, "pixel fit")
          + underbrace(lambda_"dssim" cal(L)_"SSIM", "structure")
        $

        #strong[Opacity #h(1em)]
        do not learn the background:
        $
          cal(L)_"opa" =
          - 1 / abs(Omega)
          sum_(p in Omega)
          underbrace((1 - m_"gt"(p)), #[background mask])
          dot
          underbrace(log(1 - alpha(p)), "opacity penalty")
        $

        #strong[Motion #h(1em)]
        move softly (locally rigid, global speed):
        $
          cal(L)_"dyn" =
          underbrace(cal(L)_"rigid", #[neighbouring Gaussians move coherently])
          +
          underbrace(cal(L)_"motion", #[suppresses high-velocity artifacts])
        $

        *+ dynamic regularization* $->$ better temporal consistency, but can add training cost
      ]

    ],

    // -------------------------------------------------------------------------
    // Column 2
    // -------------------------------------------------------------------------
    [

      #pop.column-box(heading: [Rendering Equation #h(1fr) #fa-icon("photo-film")])[
        Condition 4D distributions to time $t$ $=>$ 3D Gauss.\
        Project to camera plane and integrate colors.

        #strong[Sort #h(1em)]
        depth-sort and composit front-to-back:
        $
          C_p^("sort")(t, v) =
          sum_i
          underbrace(T_i (p,t), #[leftover \ light])
          underbrace(alpha_i (p,t), "opacity")
          underbrace(c_i (v,t), "color")
          + underbrace(T_(N+1)c_("bg"), #[background])
        $

        #strong[Sort-Free #h(1em)] MLP-learned blending weights:
        $
          C_p^("sf")(t, v) =
          sum_i
          underbrace(beta_i (p,t), #[MLP weight])
          c_i (v,t)
          + underbrace(beta_("bg")(p,t)c_("bg"), "background")
        $
        *+ input time t* $->$ Works in 4D, but expensive MLP
      ]

      #pop.column-box(heading: [Ablations and Evaluation #h(1fr) #fa-icon("swatchbook")])[
        #table(
          columns: (0.5fr, 1.3fr),
          inset: 4.5pt,
          stroke: (x, y) => if y > 0 { (top: 0.35pt + line-color) },
          [*Shape*], [Ellipsoid $dot.c$ Sphere],
          [*Color*], [RGB $dot.c$ SH(3)],
          [*Renderer*], [Sorting $dot.c$ Sort-free],
          [*Pruning*], [Opacity $dot.c$ Spatio-temporal | Densify?],
          [*Regularize*], [Dropout, Prune-Densify, Uncertainty],
        )
        Dynamic D-NeRF Data: *T-Rex*, *Bouncing Balls*
      ]


      #pop.column-box(
        heading: [Bottlenecks Incurred #h(1fr) #fa-icon("heart-crack")],
        heading-box-args: final-heading-args,
      )[
        #grid(
          columns: (1fr, 1fr),
          gutter: 0.65em,

          [
            #callout(
              [Sort-free in 4D],
              [Adding time to MobileGS MLPs moves bottleneck from sorting to MLP eval.],
              fill: rgb("#fff7ed"),
              accent: orange,
            )
          ],

          [
            #callout(
              [Uncertainty loss],
              [USplat-style weighting is too expensive for full ablations, so we tested small-scale.],
              fill: rgb("#f5f3ff"),
              accent: violet,
            )
          ],

          [
            #callout(
              [Sort-free + RGB],
              [Pure color does not carry information to differentiate enough depth for Gaussians.],
              fill: rgb("#efffef"),
              accent: green,
            )
          ],
        )
      ]

    ],

    // -------------------------------------------------------------------------
    // Column 3
    // -------------------------------------------------------------------------
    [
      #pop.column-box(heading: [Feature Matrix #h(1fr) #fa-icon("list")])[
        #set text(size: 0.78em)
        #contrib-table-large
      ]

      #pop.column-box(
        heading: [Main takeaway #h(1fr) #fa-icon("file-lines")],
        heading-box-args: final-heading-args,
      )[
        #underline()[No configuration dominates all metrics]
        Reconstruction, FPS, memory, train time $=>$ Fight

        #box(
          width: 100%,
          fill: rgb("#cee2f6"),
          radius: 14pt,
          inset: 14pt,
        )[
          #text(fill: black, weight: "bold", size: 1.0em)[Practical Best]
          #linebreak()
          #text(fill: black, size: 0.8em)[
            ellipsoid · RGB · sort-based · interleaved pruning · no dropout
          ]
        ]
        Quality? ellipsoid, SH(3), and sort-based rendering

        Smallest? RGB and interleaved pruning keeps checkpoints small

        Fastest? validate the renderer; sort-free is not always faster in 4D
      ]
    ],
  )
]



// -----------------------------------------------------------------------------
// BOTTOM 1/3: light blue result area
// -----------------------------------------------------------------------------
#box(
  width: 100%,
  height: bottom-h,
  fill: bottom-blue,
  radius: 20pt,
  inset: 0.65cm,
)[
  #grid(
    columns: (2fr, 1fr),
    gutter: box-spacing,

    // -------------------------------------------------------------------------
    // First two columns: 2 x 2 image/result blocks
    // -------------------------------------------------------------------------
    [
      #grid(
        columns: (1fr, 1fr),
        rows: (15.7cm, 15.7cm),
        gutter: 0.78cm,

        [
          #pop.column-box(
            heading: "Bouncing Balls",
            heading-box-args: result-heading-args,
            body-box-args: result-body-args,
          )[
            #img-slot(
              [Qualitative reconstruction grid],
              [native / OMNI / ablations],
              height: 10.1cm,
            )
          ]
        ],

        [
          #pop.column-box(
            heading: "Evaluation Result",
            heading-box-args: result-heading-args,
            body-box-args: result-body-args,
          )[
            #img-slot(
              [Metric comparison grid],
              [PSNR · SSIM · LPIPS · FPS · VRAM],
              height: 10.1cm,
            )
          ]
        ],

        [
          #pop.column-box(
            heading: "T-Rex",
            heading-box-args: result-heading-args,
            body-box-args: result-body-args,
          )[
            #img-slot(
              [Frame and crop grid],
              [sort-based / sort-free / RGB / SH(3)],
              height: 10.1cm,
            )
          ]
        ],

        [
          #pop.column-box(
            heading: "MOG: Bitter Lesson",
            heading-box-args: result-heading-args,
            body-box-args: result-body-args,
          )[
            #img-slot(
              [Motion and camera analysis],
              [fixed camera · moving camera · artifacts],
              height: 10.1cm,
            )
          ]
        ],
      )
    ],

    // -------------------------------------------------------------------------
    // Last column: deployment guide + bibliography
    // -------------------------------------------------------------------------
    [

      #pop.column-box(
        heading: [Bibliography #h(1fr) #fa-icon("book-bookmark")],
        heading-box-args: final-heading-args,
        body-box-args: final-body-args,
      )[

        #ref-line([Yang et al.], [Native 4D Gaussian Splatting backbone.])
        #ref-line([Luo et al.], [Instant4D isotropic variants and spatio-temporal pruning.])
        #ref-line([Du et al.], [MobileGS sort-free rendering and compression ideas.])
        #ref-line([Hou et al.], [Sort-Free Gaussian Splatting.])
        #ref-line([Yuan et al.], [4DGS at 1000 FPS visibility masks and pruning schedules.])
        #ref-line([Guo et al.], [Uncertainty-aware training for dynamic Gaussian splatting.])

        // If you prefer full BibTeX output and have bibliography.bib in the folder,
        // replace the manual list above with:
        // #bibliography("bibliography.bib")
      ]
    ],
  )
]
