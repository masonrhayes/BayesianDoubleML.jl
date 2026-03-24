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
    input_sha = "95f631d9a8f8f5dac736ba646093672a01ee8f75c402f61e9df89585031d34aa"
    julia_version = "1.12.5"
-->




<div class="markdown"><h1 id="BayesianDoubleML-MCMC-example">BayesianDoubleML MCMC example</h1></div>

<pre class='language-julia'><code class='language-julia'>begin
    using BayesianDoubleML
    using StableRNGs
end</code></pre>



<div class="markdown"><h3 id="Data-generation">Data generation</h3></div>


<div class="markdown"><p>Let's generate data as per Section 6 of the paper </p></div>

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


<div class="markdown"><p>We then define the <code>BDMLModel</code>, using the BDML-Hier model. </p><p>As per Section 6 of the paper, the BDML-Hier model "allows different standard deviations in the normal shrinkage priors for <span class="tex">\(\delta\)</span> and <span class="tex">\(\gamma\)</span> ... with a hierarchical prior that places independent Inverse-Gamma(2, 2) hyper-priors on <span class="tex">\(\sigma^2_\delta\)</span> and <span class="tex">\(\sigma^2_\gamma\)</span>."</p></div>

<pre class='language-julia'><code class='language-julia'>model = BDMLModel(df, :y, :d; model_type = :hier)</code></pre>
<pre class="code-output documenter-example-output" id="var-model">BDMLHierarchicalModel (not fitted)
  Observations: 200
  Covariates: 100
</pre>


<div class="markdown"><h3 id="Model-fitting">Model fitting</h3></div>


<div class="markdown"><p>We then solve the inference problem via MCMC, using the No-U-Turn Sampler (NUTS), as in the paper:</p></div>

<pre class='language-julia'><code class='language-julia'>fit!(
    model,
    MCMCMethod(:nuts),
    n_chains = 8,
    n_samples = 400
)</code></pre>



<div class="markdown"><h3 id="Model-results">Model results</h3></div>

<pre class='language-julia'><code class='language-julia'>begin
    summary(model)
    coeftable(model)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash123630">Bayesian Double ML Coefficient Table
======================================================================
Parameter: α (treatment effect)
Model type: hier
Inference method: MCMC
Credible interval level: 95.0% (HPD)
Number of posterior samples: 3200

  Parameter     Estimate   Std. Error         MCSE      P-value
  ---------     --------   ----------         ----      -------
  α               1.9170       0.1967       0.0015       0.0000

HPD Credible Intervals:
  α: [1.5347, 2.2952]

Diagnostics:
  Effective Sample Size (ESS): 1747.9
</pre>

<!-- PlutoStaticHTML.End -->
```

