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
    input_sha = "9d84e157c57bed87ead15f806f838ab6b91b6f1b30c315fa58b230b1966ea032"
    julia_version = "1.12.5"
-->




<div class="markdown"><h1 id="BayesianDoubleML-VI-example">BayesianDoubleML VI example</h1></div>

<pre class='language-julia'><code class='language-julia'>begin
    using BayesianDoubleML
    using Mooncake # explictly import Mooncake for AD
    using StableRNGs
    using PlutoUI
end</code></pre>



<div class="markdown"><h3 id="Data-generation">Data generation</h3></div>


<div class="markdown"><p>Let's generate data as per Section 6 of the paper:</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Define parameters
    n = 200
    p = 100
    alpha_true = 2.0

    rng = StableRNG(42)

    # Generate data as DataFrame
    df = make_plr_DTL2025(n, p, 2.0; alpha = alpha_true, rng = rng)
end</code></pre>
<table><tbody><tr><th></th><th>X1</th><th>X2</th><th>X3</th><th>X4</th><th>X5</th><th>X6</th><th>X7</th><th>X8</th><th>...</th></tr><tr><td>1</td><td>-0.670252</td><td>0.394702</td><td>-0.792956</td><td>0.586774</td><td>1.37908</td><td>-0.229871</td><td>-1.94115</td><td>0.562663</td><td></td></tr><tr><td>2</td><td>0.447122</td><td>-0.609501</td><td>-0.126087</td><td>0.664461</td><td>-0.930326</td><td>-1.26984</td><td>0.584906</td><td>0.911899</td><td></td></tr><tr><td>3</td><td>1.37363</td><td>0.20276</td><td>-1.48668</td><td>-0.351543</td><td>0.930427</td><td>-1.59038</td><td>0.792325</td><td>1.01552</td><td></td></tr><tr><td>4</td><td>1.30954</td><td>-1.19351</td><td>0.933002</td><td>0.467756</td><td>0.706468</td><td>2.31795</td><td>0.757444</td><td>-1.01594</td><td></td></tr><tr><td>5</td><td>0.12607</td><td>-0.0511932</td><td>0.893514</td><td>-0.925222</td><td>0.90671</td><td>-0.42725</td><td>0.197433</td><td>0.349011</td><td></td></tr><tr><td>6</td><td>0.683948</td><td>0.433951</td><td>-2.16082</td><td>-0.176606</td><td>0.729576</td><td>-1.2115</td><td>-0.28795</td><td>1.25103</td><td></td></tr><tr><td>7</td><td>-1.0192</td><td>2.0503</td><td>0.0790862</td><td>0.15015</td><td>0.0842054</td><td>-1.67432</td><td>0.0519768</td><td>0.81464</td><td></td></tr><tr><td>8</td><td>-0.793513</td><td>0.483918</td><td>-0.0155465</td><td>1.23817</td><td>1.92684</td><td>-0.385009</td><td>0.563696</td><td>0.280145</td><td></td></tr><tr><td>9</td><td>1.77472</td><td>1.39697</td><td>-0.320127</td><td>-1.877</td><td>-1.98949</td><td>0.853332</td><td>-0.858619</td><td>-0.0564358</td><td></td></tr><tr><td>10</td><td>1.29735</td><td>0.156428</td><td>-0.302891</td><td>-1.49772</td><td>0.63167</td><td>0.370054</td><td>0.749946</td><td>1.03018</td><td></td></tr><tr><td>...</td></tr><tr><td>200</td><td>0.536166</td><td>0.190568</td><td>-1.67827</td><td>1.23419</td><td>-0.738023</td><td>0.68223</td><td>0.30193</td><td>-0.530524</td><td></td></tr></tbody></table>


<div class="markdown"><h3 id="Model-setup">Model setup</h3></div>


<div class="markdown"><p>We then define the BDML model, using the BDML-Hier model. </p><p>As per Section 6 of the paper, the BDML-Hier model "allows different standard deviations in the normal shrinkage priors for <span class="tex">\(\delta\)</span> and <span class="tex">\(\gamma\)</span> ... with a hierarchical prior that places independent Inverse-Gamma(2, 2) hyper-priors on <span class="tex">\(\sigma^2_\delta\)</span> and <span class="tex">\(\sigma^2_\gamma\)</span>."</p></div>

<pre class='language-julia'><code class='language-julia'>model = BDMLModel(df, :y, :d; model_type = :hier)</code></pre>
<pre class="code-output documenter-example-output" id="var-model">BDMLHierarchicalModel (not fitted)
  Observations: 200
  Covariates: 100
</pre>


<div class="markdown"><h3 id="Model-fitting">Model fitting</h3></div>


<div class="markdown"><p>And we then solve the problem using Automatic Differentiation Variational Inference (ADVI). </p><p>In this example, we first try using the SimpleVIMethod with the AutoMooncake AD backend. (Note: AutoMooncake from <a href="https://chalk-lab.github.io/Mooncake.jl/stable/">Mooncake.jl</a> provides extremely fast automatic differentiation, at the cost of a bit longer compile time.)</p></div>

<pre class='language-julia'><code class='language-julia'>fit!(
    model,
    SimpleVIMethod(; ad_backend = AutoMooncake),
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
  α               1.0060       0.1695       0.0000       0.0000

HPD Credible Intervals:
  α: [0.7024, 1.3624]

Diagnostics:
  Final ELBO: -724.22
</pre>


<div class="markdown"><h2 id="Problems-more-suitable-to-ADVI">Problems more suitable to ADVI</h2><p>As we see above, for this problem, ADVI is <em>not</em> a good fit for the problem above where <span class="tex">\(p\)</span> is large relative to <span class="tex">\(n\)</span>; ADVI is not as able to reach a good approximation, at least not with this data generation process. The true causal effect is 2.0, but the above model estimated 1.01.</p><p>However, ADVI is yields a good approximation in a variety of other real-world scenarios; let's try a case where e.g., n=1000, p = 31.</p><p>As a general rule of thumb: in anecdotal testing, ADVI methods are generally reliable on similar problems when <span class="tex">\(n &gt;&gt; p\)</span>.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    n2 = 1_000
    lower_p = floor(sqrt(n2)) |&gt; Int
    upper_p = floor(n2 / 2) |&gt; Int
    default_p = Int(floor(sqrt(n2)))
    @assert lower_p &lt; upper_p
    @bind p2 Slider(lower_p:10:upper_p, show_value = true, default = default_p)
end</code></pre>
<bond def="p2" unique_id="uotlvbvfujde"><input max="47" min="1" type="range" value="1"/><script>
					const input_el = currentScript.previousElementSibling
					const output_el = currentScript.nextElementSibling
					const displays = ["31", "41", "51", "61", "71", "81", "91", "101", "111", "121", "131", "141", "151", "161", "171", "181", "191", "201", "211", "221", "231", "241", "251", "261", "271", "281", "291", "301", "311", "321", "331", "341", "351", "361", "371", "381", "391", "401", "411", "421", "431", "441", "451", "461", "471", "481", "491"]

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
    					display: inline-block;">31</output></bond>

<pre class='language-julia'><code class='language-julia'>begin
    # Generate data as DataFrame
    df2 = make_plr_DTL2025(n2, p2, 2.0; alpha = alpha_true, rng = rng)
end</code></pre>
<table><tbody><tr><th></th><th>X1</th><th>X2</th><th>X3</th><th>X4</th><th>X5</th><th>X6</th><th>X7</th><th>X8</th><th>...</th></tr><tr><td>1</td><td>1.55515</td><td>0.834176</td><td>0.0794264</td><td>0.816123</td><td>-0.577523</td><td>-0.258665</td><td>0.409917</td><td>-0.0552729</td><td></td></tr><tr><td>2</td><td>1.33166</td><td>-0.600332</td><td>0.452788</td><td>-2.34618</td><td>1.18973</td><td>0.184351</td><td>-0.247993</td><td>-1.99808</td><td></td></tr><tr><td>3</td><td>-0.540096</td><td>-1.41948</td><td>0.84074</td><td>-1.09164</td><td>-1.58453</td><td>-0.0189462</td><td>0.752004</td><td>0.205925</td><td></td></tr><tr><td>4</td><td>0.226415</td><td>-0.194173</td><td>-1.0547</td><td>1.26822</td><td>0.705675</td><td>-0.575521</td><td>1.67878</td><td>-0.693857</td><td></td></tr><tr><td>5</td><td>-0.765495</td><td>1.66478</td><td>-0.785242</td><td>1.24036</td><td>1.57757</td><td>-1.31501</td><td>0.806199</td><td>1.1712</td><td></td></tr><tr><td>6</td><td>-0.182184</td><td>0.745669</td><td>-1.47945</td><td>1.17658</td><td>-1.99361</td><td>-0.427153</td><td>0.39736</td><td>-0.468194</td><td></td></tr><tr><td>7</td><td>0.971872</td><td>0.513281</td><td>-0.965874</td><td>-0.185508</td><td>0.57481</td><td>0.539647</td><td>-0.154606</td><td>0.577174</td><td></td></tr><tr><td>8</td><td>0.547728</td><td>-0.631522</td><td>-2.37676</td><td>-0.508254</td><td>0.520338</td><td>0.362892</td><td>-0.482445</td><td>1.76727</td><td></td></tr><tr><td>9</td><td>-0.0327601</td><td>0.711689</td><td>-1.20202</td><td>0.383775</td><td>1.20067</td><td>0.660245</td><td>0.129591</td><td>-0.996378</td><td></td></tr><tr><td>10</td><td>-1.82501</td><td>-0.830343</td><td>-0.0752587</td><td>-0.174452</td><td>-0.811727</td><td>-0.852999</td><td>0.432393</td><td>0.596293</td><td></td></tr><tr><td>...</td></tr><tr><td>1000</td><td>0.469417</td><td>-0.119989</td><td>0.338874</td><td>1.4048</td><td>-1.14472</td><td>-0.0875275</td><td>-0.86424</td><td>1.96033</td><td></td></tr></tbody></table>

<pre class='language-julia'><code class='language-julia'>model2 = BDMLModel(df2, :y, :d; model_type = :hier)</code></pre>
<pre class="code-output documenter-example-output" id="var-model2">BDMLHierarchicalModel (not fitted)
  Observations: 1000
  Covariates: 31
</pre>

<pre class='language-julia'><code class='language-julia'>fit!(
    model2,
    SimpleVIMethod(; ad_backend = AutoMooncake),
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
  α               1.9537       0.0612       0.0000       0.0000

HPD Credible Intervals:
  α: [1.841, 2.0798]

Diagnostics:
  Final ELBO: -2152.67
</pre>

<!-- PlutoStaticHTML.End -->
```

