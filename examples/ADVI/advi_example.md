```@raw html
<style>
    #documenter-page table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    #documenter-page pre, #documenter-page div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }

    .admonition-body {
        padding: 0em 1.25em !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "5cd74ee88d55490b9b4b5cd78fe79976e317cb9c2b0eb4cdaac6d7d1cbd456e8"
    julia_version = "1.12.6"
-->




<div class="markdown"><h1 id="BayesianDoubleML-VI-example">BayesianDoubleML VI example</h1></div>

<pre class='language-julia'><code class='language-julia'>begin
    using BayesianDoubleML
    using Mooncake # explictly import Mooncake for AD
    using StableRNGs
    using PlutoUI
end</code></pre>



<div class="markdown"><h3 id="Data-generation">Data generation</h3></div>


<div class="markdown"><p>Let's generate data as per Section 6 <a href="http://arxiv.org/abs/2508.12688">DiTraglia and Liu (2025)</a>.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Define parameters
    n = 200
    p = 100
    alpha_true = 2.0

    rng = StableRNG(42)

    # Generate data as DataFrame
    df = make_plr_DTL2025(n, p, 2.0; alpha = alpha_true, rng = rng)
end;</code></pre>



<div class="markdown"><h3 id="Model-setup">Model setup</h3></div>


<div class="markdown"><p>We then define the BDMLModel, using the hierarchical model (BDML-Hier) from the paper:</p><p>As per Section 6 of the paper, the BDML-Hier model "allows different standard deviations in the normal shrinkage priors for <span class="tex">\(\delta\)</span> and <span class="tex">\(\gamma\)</span> ... with a hierarchical prior that places independent Inverse-Gamma(2, 2) hyper-priors on <span class="tex">\(\sigma^2_\delta\)</span> and <span class="tex">\(\sigma^2_\gamma\)</span>."</p></div>

<pre class='language-julia'><code class='language-julia'>model = BDMLModel(df, :y, :d; model_type = :hier)</code></pre>
<pre class="code-output documenter-example-output" id="var-model">BDMLHierarchicalModel (not fitted)
  Observations: 200
  Covariates: 100
</pre>


<div class="markdown"><h3 id="Model-fitting">Model fitting</h3></div>


<div class="markdown"><p>And we then fit the model using Automatic Differentiation Variational Inference (ADVI). </p><p>In this example, we first try using the CollapsedVI method with the AutoMooncake AD backend. (Note: AutoMooncake from <a href="https://chalk-lab.github.io/Mooncake.jl/stable/">Mooncake.jl</a> provides extremely fast automatic differentiation, at the cost of a longer compile time.)</p><p>The <code>CollapsedVI</code> method differs from the MCMC implementation in that it works by <strong>integrating out</strong> the large coefficient vectors before variational optimization, making it magnitudes faster while retaining good inference. It does this using <a href="https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem">Rao–Blackwellization</a> of the estimator of the causal parameter.</p></div>

<pre class='language-julia'><code class='language-julia'>fit!(
    model,
    CollapsedVI(; ad_backend = AutoMooncake),
    n_iterations = 1_000,
    show_progress = false
);</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    summary(model)
    coeftable(model)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash123630">Bayesian Double ML Coefficient Table
======================================================================
Parameter: α (treatment effect)
Model type: hier
Inference method: VI
Credible interval level: 95.0% (HPD)
Number of posterior samples: 2000

  Parameter     Estimate   Std. Error         MCSE      P-value
  ---------     --------   ----------         ----      -------
  α               1.9637       0.1955       0.0000       0.0000

HPD Credible Intervals:
  α: [1.608, 2.3592]

Diagnostics:
  Final ELBO: -667.07
</pre>


<div class="markdown"><h2 id="Problems-less-suitable-to-ADVI">Problems less suitable to ADVI</h2><p>As we see above, for this problem, ADVI is a good fit for the problem above where <span class="tex">\(p\)</span> is large relative to <span class="tex">\(n\)</span>; ADVI is able to reach a good approximation, at least with this data generation process. The true causal effect is 2.0, and the above model estimated 1.96.</p><p>However, ADVI does not yield a good approximation where <span class="tex">\(n &lt;&lt; p\)</span>.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    n2 = 50
    lower_p = floor(n2) |&gt; Int
    upper_p = floor(4 * n2) |&gt; Int
    default_p = Int(floor(n2 * 2))
    @assert lower_p &lt; upper_p
    @bind p2 Slider(lower_p:10:upper_p, show_value = true, default = default_p)
end</code></pre>
<bond def="p2" unique_id="riqjuvlgnwzt"><input max="16" min="1" type="range" value="6"/><script>
					const input_el = currentScript.previousElementSibling
					const output_el = currentScript.nextElementSibling
					const displays = ["50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "150", "160", "170", "180", "190", "200"]

					let update_output = () => {
						output_el.value = displays[input_el.valueAsNumber - 1]
					}
					
					input_el.addEventListener("input", update_output)
					// We also poll for changes because the `input_el.value` can change from the outside, e.g. https://github.com/JuliaPluto/PlutoUI.jl/issues/277
					let id = setInterval(update_output, 200)
					invalidation.then(() => {
						clearInterval(id)
						input_el.removeEventListener("input", update_output)
					})
					</script><output style="
						font-family: system-ui;
						font-variant-numeric: tabular-nums;
    					font-size: 15px;
    					margin-left: 3px;
    					transform: translateY(-4px);
    					display: inline-block;">100</output></bond>

<pre class='language-julia'><code class='language-julia'>begin
    # Generate data as DataFrame
    df2 = make_plr_DTL2025(n2, p2, 2.0; alpha = alpha_true, rng = rng)
end;</code></pre>


<pre class='language-julia'><code class='language-julia'>model2 = BDMLModel(df2, :y, :d; model_type = :hier)</code></pre>
<pre class="code-output documenter-example-output" id="var-model2">BDMLHierarchicalModel (not fitted)
  Observations: 50
  Covariates: 100
</pre>

<pre class='language-julia'><code class='language-julia'>fit!(
    model2,
    CollapsedVI(; ad_backend = AutoMooncake),
    n_iterations = 1_000,
    show_progress = false
);</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    summary(model2)
    coeftable(model2)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash178616">Bayesian Double ML Coefficient Table
======================================================================
Parameter: α (treatment effect)
Model type: hier
Inference method: VI
Credible interval level: 95.0% (HPD)
Number of posterior samples: 2000

  Parameter     Estimate   Std. Error         MCSE      P-value
  ---------     --------   ----------         ----      -------
  α               0.1167       2.3469       0.0000       0.9320

HPD Credible Intervals:
  α: [-4.5729, 3.6111]

Diagnostics:
  Final ELBO: -230.6
</pre>

<!-- PlutoStaticHTML.End -->
```

