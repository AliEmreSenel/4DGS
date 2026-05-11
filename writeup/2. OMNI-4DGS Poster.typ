#import "@preview/peace-of-posters:0.6.0" as pop
#import "@preview/fontawesome:0.6.0": *
#import "appendix.typ": contrib-table-large

#set page("a0", margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)

#set text(size: pop.layout-a0.at("body-size"))
#set par(justify: false, leading: 0.5em)

#let box-spacing = 1.2em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)


#let result-img-width = 9.2cm

// -----------------------------------------------------------------------------
// Total content height (title box excluded) — kept for v(1fr) anchor
// -----------------------------------------------------------------------------
#let content-h = 105cm

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
#let box-height = auto
#let top-space = 8pt

#let result-heading-args = pop.uni-fr.heading-box-args
#result-heading-args.insert("fill", rgb("#2343a4"))
#result-heading-args.insert("height", box-height)
#result-heading-args.insert("inset", (
  top: top-space,
  right: 0.6em,
  bottom: 0.6em,
  left: 0.6em,
))

#let result-body-args = pop.uni-fr.body-box-args
#result-body-args.insert("fill", white)
#result-body-args.insert("height", box-height)
#result-body-args.insert("inset", (
  top: top-space,
  right: 18pt,
  bottom: 18pt,
  left: 18pt,
))

#let final-heading-args = pop.uni-fr.heading-box-args
#final-heading-args.insert("fill", navy)
#final-heading-args.insert("height", box-height)
#final-heading-args.insert("inset", (
  top: top-space,
  right: 0.6em,
  bottom: 0.6em,
  left: 0.6em,
))

#let final-body-args = pop.uni-fr.body-box-args
#final-body-args.insert("fill", white)
#final-body-args.insert("height", box-height)
#final-body-args.insert("inset", (
  top: top-space,
  right: 18pt,
  bottom: 18pt,
  left: 18pt,
))

#let std-heading-args = pop.uni-fr.heading-box-args
#std-heading-args.insert("height", box-height)
#std-heading-args.insert("inset", (top: top-space, right: 0.6em, bottom: 0.6em, left: 0.6em))

// -----------------------------------------------------------------------------
// Title
// -----------------------------------------------------------------------------
#let title-grid-box(
  title,
  subtitle: none,
  authors: none,
  institutes: none,
  keywords: none,
  right-body: none,
  logo: none,
  background: none,
  text-relative-width: 62%,
  spacing: 5%,
  title-size: none,
  subtitle-size: none,
  authors-size: none,
  institutes-size: none,
  keywords-size: none,
  right-area-height: 7.2em,
) = context {
  let text-relative-width = text-relative-width

  let pt = pop._state-poster-theme.get()
  let pl = pop._state-poster-layout.get()

  let title-size = if title-size == none { pl.at("title-size") } else { title-size }
  let subtitle-size = if subtitle-size == none { pl.at("subtitle-size") } else { subtitle-size }
  let authors-size = if authors-size == none { pl.at("authors-size") } else { authors-size }
  let institutes-size = if institutes-size == none { pl.at("institutes-size") } else { institutes-size }
  let keywords-size = if keywords-size == none { pl.at("keywords-size") } else { keywords-size }

  let right-content = if right-body != none { right-body } else { logo }

  let title-text = [
    #set text(size: title-size)
    #title\
  ]

  let lower-text = [
    #set text(size: subtitle-size)
    #if subtitle != none { [#subtitle\ ] }

    #v(0.35em, weak: true)

    #set text(size: authors-size)
    #if authors != none { [#authors\ ] }

    #if institutes != none {
      [
        #set text(size: institutes-size)
        #institutes
      ]
    }

    #if keywords != none {
      [
        #v(0.45em, weak: true)
        #set text(size: keywords-size)
        #keywords
      ]
    }
  ]

  if right-content == none {
    text-relative-width = 100%
  }

  let title-box-args = pt.at(
    "title-box-args",
    default: pt.at("heading-box-args", default: ()),
  )

  let title-text-args = pt.at(
    "title-text-args",
    default: pt.at("heading-text-args", default: ()),
  )

  let title-box-function = pt.at(
    "title-box-function",
    default: rect,
  )

  let title-content = box(width: 100%)[
    #box(width: 100%)[
      #title-text
    ]

    #box(width: text-relative-width)[
      #lower-text
    ]

    #if right-content != none {
      place(top + right)[
        #box(
          width: 100% - text-relative-width,
          height: right-area-height,
        )[
          #right-content
        ]
      ]
    }
  ]

  pop.common-box(
    heading: [
      #if background != none {
        [
          #background
          #v(-measure(background).height)
        ]
      }
      #title-content
    ],
    heading-box-args: title-box-args,
    heading-text-args: title-text-args,
    heading-box-function: title-box-function,
  )
}

#title-grid-box(
  [OMNI-4DGS: Fast, Light and Precise Video-To-Model Reconstruction],
  authors: "Ali Emre Senel¹ · Tebe Nigrelli¹ · Stefana Chiriac¹",
  institutes: [¹Bocconi University, Milan, Italy],
  keywords: "4D Gaussian Splatting · Dynamic Scene Reconstruction · Ablations · Efficient Rendering",

  text-relative-width: 80%,
  spacing: 0%,
  right-area-height: 7.2em,

  right-body: [
    #box(width: 100%, height: 80%)[
      #set text(size: 0.7em)

      #place(top + right)[
        #text(weight: "semibold")[2026-05-12]
      ]

      #place(top + right, dy: 3.5em)[
        #text(weight: "bold")[Università Bocconi] \ Milano
      ]

      #place(bottom + right)[
        #fa-icon("github")
        #h(0.25em)
        #text(weight: "medium")[AliEmreSenel/4DGS]
      ]
    ]
  ],
)
// -----------------------------------------------------------------------------
// Full-height single 3-column layout
// -----------------------------------------------------------------------------
#box(width: 100%, height: content-h)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    rows: (content-h,),
    gutter: box-spacing,

    // -------------------------------------------------------------------------
    // Column 1
    // -------------------------------------------------------------------------
    [
      #pop.column-box(
        heading: [Context and Motivation #h(1fr) #fa-icon("map", solid: true)],
        heading-box-args: std-heading-args,
      )[
        4D Gaussian Splatting (4DGS) is a leading approach to dynamic scene reconstruction from video.

        _Fast and flexible_ $=>$ But not optimized!

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

      #pop.column-box(
        heading: [4D Gaussian Building Blocks #h(1fr) #fa-icon("puzzle-piece")],
        heading-box-args: std-heading-args,
      )[
        Each primitive is centered in space-time and stores opacity, shape, rotation, and color. The native representation is expressive, but compact variants trade quality for speed and size.

        $
          G_i = (mu_i, Sigma_i, o_i, r_i, arrow("SH")_i),
          quad mu_i in RR^4,
          quad Sigma_i in RR^(4 times 4)
        $


        #table(
          columns: (0.75fr, 0.3fr, 2fr),
          inset: 10pt,
          align: (x, y) => if x == 1 { center } else { left },
          stroke: (x, y) => if y > 0 { (top: 0.35pt + line-color) },

          [*4D mean*], [$mu_i$], [mean is the gaussian center],
          [*Covariance*], [$Sigma_i$], [(sphere || ellipsoid) $=:$ *Isotropy*],
          [*Opacity*], [$o_i$], [adjusts visibility / see-through],
          [*Color*], [$"SH"_i$], [$"SH"(0) = "RGB" arrow.r.long.squiggly "SH"(3)$],
        )
      ]

      #pop.column-box(
        heading: [Ablations and Evaluation #h(1fr) #fa-icon("swatchbook")],
        heading-box-args: std-heading-args,
      )[
        #table(
          columns: (0.5fr, 1.3fr),
          inset: 4.5pt,
          stroke: (x, y) => if y > 0 { (top: 0.35pt + line-color) },
          [*Shape*], [Ellipsoid $dot.c$ Sphere],
          [*Color*], [RGB $dot.c$ SH(3)],
          [*Renderer*], [Sorting $dot.c$ Sort-Free],
          [*Pruning*], [Opacity $dot.c$ Spatio-Temporal],
          [*Densify*], [Edge-Guided, Loss Guided],
          [*Regularize*], [Dropout, Prune-Densify, Uncertainty],
        )
        Dynamic D-NeRF Data: *T-Rex*, *Bouncing Balls*
      ]

      #pop.column-box(
        heading: [Bouncing Balls #h(1fr) #fa-icon("baseball")],
        heading-box-args: result-heading-args,
      )[
        *Fixed*: No USplat/Prune/ESS/Dropout, Sort, 10k.
        #grid(
          columns: (0.1fr, 1fr, 1fr),
          rows: (auto, auto, auto),
          column-gutter: 5pt,
          row-gutter: 5pt,
          align: center + horizon,

          [], [#strong[Sort]], [#strong[Sort-Free]],

          [#rotate(-90deg)[#strong[Ellipsoid]]],
          [#image(
            "./img/bouncingballs__anisotropic__no_usplat__sh3__sort__no_pruning__no_dropout__no_ess__10000.png",
            width: result-img-width,
          )],
          [#image(
            "./img/bouncingballs__anisotropic__use_usplat__sh3__sort_free__no_pruning__no_dropout__no_ess__10000.png",
            width: result-img-width,
          )],

          [#rotate(-90deg)[#strong[Spherical]]],
          [#image(
            "./img/bouncingballs__isotropic__no_usplat__sh3__sort__no_pruning__no_dropout__no_ess__10000.png",
            width: result-img-width,
          )],
          [#image(
            "./img/bouncingballs__isotropic__use_usplat__sh3__sort_free__no_pruning__no_dropout__no_ess__10000.png",
            width: result-img-width,
          )],
        )
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
              [Sort-Free in 4D],
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
        )
      ]
      // Result boxes — col 1

      #pop.column-box(
        heading: [Best Ablation Results #h(1fr) #fa-icon("crown")],
        heading-box-args: final-heading-args,
      )[
        #set table(
          stroke: 0.25pt + black,
          inset: (x: 4pt, y: 5pt),
          align: center,
        )

        Best Ablations: *aniso · SH3 · sort · no-prune*

        #table(
          columns: (0.5fr, 0.45fr, 0.45fr, 0.45fr, 0.35fr, 0.45fr, 0.45fr, 0.45fr),
          inset: 10pt,
          align: (x, y) => if x > 0 { center } else { left },
          stroke: (x, y) => if y > 0 { (top: 0.35pt + line-color) },

          text(size: 0.8em)[*Scene*],
          text(size: 0.6em)[*PSNR ↑*],
          text(size: 0.6em)[*SSIM ↑*],
          text(size: 0.6em)[*LPIPS ↓*],
          text(size: 0.6em)[*FPS ↑*],
          text(size: 0.6em)[*MB ↓*],
          text(size: 0.6em)[*Gauss ↓*],
          text(size: 0.6em)[*Train ↓*],

          [BBalls], [33.39], [0.982], [0.014], [290], [278], [150k], [4.9m],

          [TRex], [30.63], [0.983], [0.023], [341], [301], [162k], [4.9m],
        )

        #align(left)[
          #text(size: 25pt)[
            Our Compute constraints $=>$ Ablations rendered at $200 times 200$.

            Direct Comparison with baselines is not possible: $400 times 400$.
          ]
        ]
      ]
    ],

    // -------------------------------------------------------------------------
    // Column 2
    // -------------------------------------------------------------------------
    [
      #pop.column-box(heading: [Rendering Equation #h(1fr) #fa-icon("photo-film")], heading-box-args: std-heading-args)[
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

        #let mhl(content) = box(
          fill: yellow.lighten(70%),
          inset: 0.10em,
          outset: 1pt,
          radius: 2pt,
          text(size: 22pt, content),
        )

        #strong[Sort-Free #h(1em)] MLP-learned blending weights:
        #[
          #show math.equation: set text(size: 28pt)

          $
            w_i (p, t, v) =
            phi_i^2(p, t, v)
            + phi_i (p, t, v) / (d_i^2(p, t, v))
            + exp(s_(max, i) / (d_i (p, t, v))) \
          $
          $
            C_p^("sf")(t, v) =
            (1 - T_p (t, v)) & frac(
                                 sum_i c_i (v, t) #mhl[$alpha_i (p, t, v) w_i (p, t, v)$],
                                 sum_i #mhl[$alpha_i (p, t, v) w_i (p, t, v)$]
                               ) \
                             & quad + T_p (t, v) c_("bg")
          $
        ]
        *+ input time t* $->$ Works in 4D, but expensive MLP
      ]

      #v(1fr)

      #pop.column-box(heading: [Training Objective #h(1fr) #fa-icon("route")], heading-box-args: std-heading-args)[
        By Ablation: fidelity, background, 4D motion.

        #strong[Photometric #h(1em)]
        Reconstruct each train view:
        $
          cal(L)_"rgb" =
          underbrace((1 - lambda_"dssim") cal(L)_1, "pixel fit")
          + underbrace(lambda_"dssim" cal(L)_"SSIM", "structure")
        $

        #strong[Opacity #h(1em)]
        Do not learn the background:
        $
          cal(L)_"opa" =
          - 1 / abs(Omega)
          sum_(p in Omega)
          underbrace((1 - m_"gt"(p)), #[background mask])
          dot
          underbrace(log(1 - alpha(p)), "opacity penalty")
        $

        #strong[Motion #h(1em)]
        Move softly (locally rigid, global speed):
        $
          cal(L)_"dyn" =
          underbrace(cal(L)_"rigid", #[near Gaussians \ move similarly])
          +
          underbrace(cal(L)_"motion", #[suppress high-velocity \ artifacts])
        $

        *+ dynamic regularization* $->$ better temporal consistency, but more expensive
      ]

      #v(1fr)

      #pop.column-box(
        heading: [TRex #h(1fr) #fa-icon("dragon")],
        heading-box-args: result-heading-args,
      )[
        *Fixed*: No USplat/Prune/ESS/Dropout, Sort, 20k.

        #grid(
          columns: (0.1fr, 1fr, 1fr),
          rows: (auto, auto, auto),
          column-gutter: 10pt,
          row-gutter: 10pt,
          align: center + horizon,

          [], [#strong[SH(3)]], [#strong[RGB]],

          [#rotate(-90deg)[#strong[Ellipsoid]]],
          [#image(
            "./img/trex__anisotropic__no_usplat__sh3__sort__no_pruning__no_dropout__no_ess__20000.png",
            width: result-img-width,
          )],
          [#image(
            "./img/trex__anisotropic__no_usplat__rgb__sort__no_pruning__no_dropout__no_ess__20000.png",
            width: result-img-width,
          )],

          [#rotate(-90deg)[#strong[Spherical]]],
          [#image(
            "./img/trex__isotropic__no_usplat__sh3__sort__no_pruning__no_dropout__no_ess__20000.png",
            width: result-img-width,
          )],
          [#image(
            "./img/trex__isotropic__no_usplat__rgb__sort__no_pruning__no_dropout__no_ess__20000.png",
            width: result-img-width,
          )],
        )
      ]

      #v(1fr)

      #pop.column-box(
        heading: [Ablation Results #h(1fr) #fa-icon("flask")],
        heading-box-args: final-heading-args,
      )[
        #set text(size: 0.8em)

        #box(
          width: auto,
          fill: rgb("#fff7ed"),
          stroke: 0.8pt + orange,
          radius: 10pt,
          inset: 9pt,
        )[
          Cohen's Coefficient $g > 0.5 => "significant"$
        ]

        #let pct(fill, body) = box(width: 3.6em)[
          #align(right)[#text(fill: fill, weight: "bold")[#body]]
        ]
        #let summary-cell(body) = table.cell(
          colspan: 6,
          inset: (top: 0pt, bottom: 20pt, left: 0pt, right: 0pt),
        )[#body]

        #table(
          columns: (5em, 1fr),
          inset: (x: 5pt, y: 8pt),
          align: left + top,
          stroke: (x, y) => (
            top: if y > 0 { 0.5pt + black } else { none },
            right: if x == 0 { 0.5pt + black } else { none },
          ),

          [
            *RGB* \
            #text(fill: green, weight: "bold")[$+0.89$]
          ],
          [
            #table(
              columns: (1fr, 4em, 1fr, 4em, 1fr, 4em),
              inset: 0pt,
              column-gutter: 8pt,
              row-gutter: 4pt,
              stroke: none,
              align: (x, y) => if x in (1, 3, 5) { right + top } else { left + top },

              summary-cell[Smaller, lighter, faster; lower quality],

              [Checkpoint], [#pct(green)[$-93%$]], [VRAM], [#pct(green)[$-79%$]], [Gaussians], [#pct(green)[$-33%$]],

              [Train], [#pct(green)[$-78%$]], [FPS], [#pct(green)[$+79%$]], [], [],
            )
          ],

          [
            *SH(3)* \
            #text(fill: red, weight: "bold")[$-0.89$]
          ],
          [
            #table(
              columns: (1fr, 4em, 1fr, 4em, 1fr, 4em),
              inset: 0pt,
              column-gutter: 8pt,
              row-gutter: 4pt,
              stroke: none,
              align: (x, y) => if x in (1, 3, 5) { right + top } else { left + top },

              summary-cell[Better quality, but expensive],

              [PSNR], [#pct(green)[$+32%$]], [LPIPS], [#pct(green)[$-83%$]], [SSIM], [#pct(green)[$+6.3%$]],
            )
          ],

          [
            *Sort* \
            #text(fill: green, weight: "bold")[$+0.84$]
          ],
          [
            #table(
              columns: (1fr, 4em, 1fr, 4em, 1fr, 4em),
              inset: 0pt,
              column-gutter: 8pt,
              row-gutter: 4pt,
              stroke: none,
              align: (x, y) => if x in (1, 3, 5) { right + top } else { left + top },

              summary-cell[Strong practical winner on our dataset],

              [LPIPS], [#pct(green)[$-85%$]], [FPS], [#pct(green)[$+290%$]], [PSNR], [#pct(green)[$+29%$]],

              [SSIM], [#pct(green)[$+8.2%$]], [Train], [#pct(green)[$-85%$]], [VRAM], [#pct(green)[$-37%$]],
            )
          ],

          [
            *Sort-Free* \
            #text(fill: red, weight: "bold")[$-0.84$]
          ],
          [
            #table(
              columns: (1fr, 4em, 1fr, 4em, 1fr, 4em),
              inset: 0pt,
              column-gutter: 8pt,
              row-gutter: 4pt,
              stroke: none,
              align: (x, y) => if x in (1, 3, 5) { right + top } else { left + top },

              summary-cell[Fewer Gaussians, but worse overall],

              [Gaussians], [#pct(green)[$-39%$]], [LPIPS], [#pct(red)[$+570%$]], [FPS], [#pct(red)[$-75%$]],

              [PSNR], [#pct(red)[$-23%$]], [SSIM], [#pct(red)[$-7.6%$]], [Train], [#pct(red)[$+590%$]],

              [VRAM], [#pct(red)[$+60%$]], [], [], [], [],
            )
          ],
        )     ]
    ],

    // -------------------------------------------------------------------------
    // Column 3
    // -------------------------------------------------------------------------
    [
      #pop.column-box(heading: [Feature Matrix #h(1fr) #fa-icon("list")], heading-box-args: std-heading-args)[
        #set text(size: 0.78em)
        #contrib-table-large
      ]

      #pop.column-box(
        heading: [MOG'D: The Bitter Lesson #h(1fr)
          #box(radius: 8pt, inset: 10pt, stroke: 0.05em + white)[
            #text(size: 20pt)[*Where is*\ *Tebe?*]
          ]],
        heading-box-args: result-heading-args,
      )[
        #grid(
          columns: (1fr, 1fr),
          gutter: 0.5cm,
          [
            #align(center)[
              #box(width: 90%)[
                #image("./img/mog_moving_cameras.png", width: 100%)
                #align(center)[5 x Moving Cameras]
              ]
            ]
          ],
          [
            #align(center)[
              #box(width: 90%)[
                #image("./img/mog_still_cameras.png", width: 100%)
                #align(center)[5 x Still Cameras]
              ]
            ]
          ],
        )

        Random Init $=>$ Need _Moving Cameras_ for artifacts.

        Data and CUDA compatibility: the real bottlenecks.
      ]

      #pop.column-box(
        heading: [Main takeaway #h(1fr) #fa-icon("file-lines")],
        heading-box-args: final-heading-args,
      )[
        #underline()[No configuration dominates all metrics]

        #box(
          width: 100%,
          fill: rgb("#cee2f6"),
          stroke: 1pt + rgb("#425161"),
          radius: 14pt,
          inset: 14pt,
        )[
          #text(fill: black, weight: "bold", size: 1.0em)[Best Practical:]
          #linebreak()
          #text(fill: black, size: 0.8em)[
            ellipsoid · RGB · sort-based · interleaved pruning · no dropout
          ]
        ]
        #v(-20pt)
        #table(
          columns: (1fr, auto),
          inset: (x: 1.5pt, y: 2.5pt),
          column-gutter: 2pt,
          align: left,
          stroke: (x, y) => if y > 0 { (top: 0.35pt + line-color) },

          [*Quality*], [ellipsoid $dot.c$ SH(3) $dot.c$ sort-based rendering],
          [*Smallest*], [RGB $dot.c$ interleaved pruning],
          [*Fastest*], [sort-based in experiments (faster)],
        )
        #v(-20pt)

        #table(
          columns: (auto, 1fr),
          inset: 0pt,
          column-gutter: 10pt,
          stroke: none,
          align: top,

          [
            #table(
              columns: (auto, auto, auto, auto),
              inset: (x: 4.5pt, y: 3.5pt),
              align: (x, y) => if x == 0 or x == 3 { center } else { left },
              stroke: (x, y) => if y > 0 { (top: 0.35pt + line-color) },

              table.cell(rowspan: 7, align: center + horizon)[
                #move(dy: 10pt)[
                  #rotate(-90deg, reflow: true)[
                    #text(weight: "bold", size: 1em)[Single effect]
                  ]
                ]
              ],
              [], [], [*Cohen g*],

              [Smallest], [RGB], [$4.91$],
              [Eval VRAM], [RGB], [$3.66$],
              [PSNR], [SH(3)], [$1.50$],
              [LPIPS], [Sort], [$1.46$],
              [Render FPS], [Sort], [$1.34$],
              [Training time], [Sort], [$1.01$],
            )
          ],

          table.cell(align: bottom)[
            #box(
              width: 100%,
              fill: rgb("#cee2f6"),
              stroke: 1pt + rgb("#425161"),
              radius: 10pt,
              inset: 9pt,
            )[
              #text(size: 0.72em)[
                RGB $=>$ compactness and eval VRAM. \
                SH(3) $=>$ +PSNR. \
                Sorted $=>$ better LPIPS, FPS, and train time.
              ]
            ]
          ],
        )
      ]

      #pop.column-box(
        heading: [Bibliography #h(1fr) #fa-icon("book-bookmark")],
        heading-box-args: final-heading-args,
      )[
        #ref-line([Kerbl et al.], [3D Gaussian Splatting for Real-Time Radiance Field Rendering.])
        #ref-line([Yang et al.], [Native 4D Gaussian Splatting backbone.])
        #ref-line([Luo et al.], [Instant4D isotropic variants and Spatio-Temporal pruning.])
        #ref-line([Du et al.], [MobileGS Sort-Free rendering and compression ideas.])
        #ref-line([Hou et al.], [Sort-Free Gaussian Splatting.])
        #ref-line([Yuan et al.], [4DGS at 1000 FPS visibility masks and pruning schedules.])
        #ref-line([Guo et al.], [Uncertainty-aware training for dynamic Gaussian splatting.])
      ]
    ],
  )
]
