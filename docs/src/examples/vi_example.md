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
    input_sha = "fd1972166cb7995656109da4f250e330ce15b2a904f22a8e4c4486ba43c351a6"
    julia_version = "1.12.5"
-->







<div class="markdown"><h1 id="BayesianDoubleML-VI-example">BayesianDoubleML VI example</h1></div>

<pre class='language-julia'><code class='language-julia'>begin
    using BayesianDoubleML
    using StableRNGs
    using PlutoUI
    using Mooncake # explictly import Mooncake for AD
end</code></pre>



<div class="markdown"><h3 id="Data-generation">Data generation</h3></div>


<div class="markdown"><p>Let's generate data as per Section 6 of the paper:</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Define parameters
    n = 200
    p = 100
    alpha_true = 2.0

    rng = StableRNG(42)

    # Generate data
    Y, D, X = generate_dgp_table1(n, p, 2.0; alpha_true = alpha_true, rng = rng)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-p">([1.6659689953789019, -1.696540541388006, -2.3269636695447327, -2.7621297542912515, -4.814719532742647, -4.764393626173836, 7.186495683831916, 4.857285589426433, 0.9995023890611519, 1.117746336470181  …  -2.6075672002159744, -0.30685854956000747, 6.4232788682865944, -7.248212253911795, 4.211637064679749, 2.534992791958787, 2.7570582368589487, 0.39241374690385733, -0.9071022595371903, -3.216234251985087], [-0.5550741592082938, 0.10454898087147146, -0.2524609343197339, -1.5524404601309103, -1.6491129047186854, -1.3018452245107413, 4.395549833064997, 3.105207044520989, 0.6365889427893981, -0.5289492581105404  …  0.11567232630969418, -0.01203588402811917, 2.9250818060095165, -3.2577780666724, 1.7939322039887973, 1.1066932520837174, 1.1365811003082429, 0.9337461214823575, 0.5669931202320483, -0.6501512252027575], [-0.6702516921145671 0.3947023567888983 … 0.17670727098731334 -0.7639873674530974; 0.4471218424633827 -0.6095014571296874 … 0.1371281741092964 0.7202850637431731; … ; -0.23654906779008752 -0.3393976643488181 … -0.1044731212869534 0.5510724485708905; 0.5361659973950985 0.19056820127614754 … 0.13987950996246476 0.2983304085843374], 2.0, (gamma = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1  …  0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], beta = [-0.1270727522393095, -0.14911624706090104, 0.008093632169739406, -0.17185187625173448, -0.02884115819645782, 0.015424000549390582, -0.047223873795245314, -0.08481805087250158, -0.17374789370936, -0.0426029531192422  …  -0.15358395534387248, -0.08667585783550474, -0.15180281621203157, -0.09050684566915979, -0.09590459147712317, -0.09140380119633015, -0.0994372177622064, 0.11792627576039201, -0.12585121572687247, -0.08195978332276374], mu_beta = [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05  …  -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05], sigma2_beta = 0.01, V = [0.3660543954470805, 0.7856920126840542, -0.6842795331415229, -1.3923604321018426, -0.420047485823469, -0.5448963814649526, 2.040578233483549, 1.4369306532101074, 1.0691804514387877, -1.1662136707920938  …  0.2222922602554177, 0.7634180708336755, 1.6546876793762253, -1.9636439091938103, 0.5561151772450789, -0.17546897469242975, 0.6807541793693491, 1.5911428852600789, 0.24512699590090478, -1.3934981512637397], epsilon = [2.046296148177815, -1.4723448564204853, -1.4575651622557146, -0.6802076641594359, -1.7622362289191877, -1.8993914580124054, -1.0635467307537136, 0.11391238195489635, -0.9757396727343455, 1.205375257696177  …  -2.202307255779401, 0.21987070102702302, 1.1150775839290048, -1.3555691631871132, 1.0655818723304393, 0.3179295239787937, 0.28597270365752087, -2.3730375800871846, -2.087002376096107, -1.5750773553079231]))</pre>


<div class="markdown"><h3 id="Model-setup">Model setup</h3></div>


<div class="markdown"><p>We then define the BDML model, using the BDML-Hier model. </p><p>As per Section 6 of the paper, the BDML-Hier model "allows different standard deviations in the normal shrinkage priors for <span class="tex">\(\delta\)</span> and <span class="tex">\(\gamma\)</span> ... with a hierarchical prior that places independent Inverse-Gamma(2, 2) hyper-priors on <span class="tex">\(\sigma^2_\delta\)</span> and <span class="tex">\(\sigma^2_\gamma\)</span>."</p></div>

<pre class='language-julia'><code class='language-julia'>model = BDMLModel(Y, D, X, model_type = :hier)</code></pre>
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
    summary(model);
    coeftable(model)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash115066">Bayesian Double ML Coefficient Table
======================================================================
Parameter: α (treatment effect)
Model type: hier
Inference method: VI
Credible interval level: 95.0% (HPD)
Number of posterior samples: 2000

  Parameter     Estimate   Std. Error         MCSE      P-value
  ---------     --------   ----------         ----      -------
  α               1.0310       0.1718       0.0000       0.0000

HPD Credible Intervals:
  α: [0.6961, 1.3535]

Diagnostics:
  Final ELBO: -740.18
</pre>


<div class="markdown"><h2 id="Problems-more-suitable-to-ADVI">Problems more suitable to ADVI</h2><p>As we see above, for this problem, ADVI is <em>not</em> a good fit for the problem above where <span class="tex">\(p\)</span> is large relative to <span class="tex">\(n\)</span>; ADVI is not as able to reach a good approximation, at least not with this data generation process. The true causal effect is 2.0, but the above model estimated 1.03.</p><p>However, ADVI is yields a good approximation in a variety of other real-world scenarios; let's try a case where e.g., n=1000, p = 31.</p><p>As a general rule of thumb: in anecdotal testing, ADVI methods are generally reliable on similar problems when <span class="tex">\(n &gt;&gt; p\)</span>.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    n2 = 1_000
    lower_p = floor(sqrt(n2)) |&gt; Int
    upper_p = floor(n2 / 2) |&gt; Int
    default_p = Int(floor(sqrt(n2)))
    @assert lower_p &lt; upper_p
    @bind p2 Slider(lower_p:10:upper_p, show_value = true, default = default_p)
end</code></pre>
<bond def="p2" unique_id="qcmoukmbcagh"><input max="47" min="1" type="range" value="1"/><script>
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
    # Generate data
    Y2, D2, X2 = generate_dgp_table1(n2, p2, 2.0; alpha_true = alpha_true, rng = rng)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-Y2">([-0.6636065984482888, -2.792398566371155, -0.08888894851760323, 1.722665486448609, 8.828404559753073, -1.892929660184188, -3.025677094624294, -3.729402992409076, 0.6419081838943952, 1.4705794954938503  …  -4.969269989036015, -1.5168533123697847, 0.22233664090062666, 1.475373662722968, 0.7090427047160928, -0.46382765683890437, 5.501035595831382, 0.7392226453188961, -5.485866594215818, 0.055518305863131], [-0.9496671142121313, -1.3150169317775937, -0.10527027098369846, 0.4876403247642984, 2.9938100451735257, -0.8882051654159393, 0.4424081141968349, -1.5519276767371437, 0.6289358535149898, -0.4382824923793728  …  -1.8332586453411697, -1.497791241289973, -1.0019250370477046, 0.04022004292708925, -1.9810396411585762, 1.936542460457599, 1.2586220499030532, -0.2658250340464889, -0.6661886701275799, -1.0178841796414742], [1.5551525214085242 0.8341761687440767 … -0.8351750796604122 -1.7332294767159886; 1.3316573018119566 -0.6003322778983055 … 1.4935892946448317 0.4209064698880064; … ; 0.4458399803650243 -1.700256412260607 … -0.28300413267577396 0.0010113671979989254; 0.469416893912413 -0.1199894621320986 … -0.41752293486355496 1.3637325262615803], 2.0, (gamma = [0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749  …  0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749, 0.1796053020267749], beta = [-0.21679797588306354, -0.006660051255094709, -0.028839987653916557, 0.11496570443839456, 0.021122553876118888, -0.2652916175898408, -0.19168480319382036, 0.06317271899422373, 0.0623725087628474, -0.2625203490423683  …  -0.09318238775876188, 0.22793915682089239, -0.16226030246164447, -0.039824182984132046, -0.4194012769822057, -0.15765288666060562, -0.04330757413251263, -0.23432647697922465, -0.28101129294794447, -0.1375890068894609], mu_beta = [-0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746  …  -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746, -0.08980265101338746], sigma2_beta = 0.03225806451612903, V = [0.4877903605986629, -1.4772134405477373, 0.5221919969046191, 0.5739260293524232, 0.9223105420265182, 0.26865518551638695, 0.4475033120708826, -1.7192825237078528, -0.17508897244630234, 0.40071467789941717  …  -0.8167568728476604, -0.6952833674030402, 0.39497386506036514, 0.9334797452030815, -1.9497738477364406, 0.47756571490746125, 0.31028451667242557, 0.4766303173167166, 0.36052329292658936, -0.7438612689696911], epsilon = [1.01446412339052, 1.8132385664633581, 0.1080134850294103, 0.6034214261185102, 3.4733810826214184, -0.527797319765279, -2.7124643805827944, -0.1170112819447805, 0.7772813582673457, 0.7854876784826555  …  -2.5954409459246426, 1.7644114638837765, 1.6418539550609292, 0.9951825432858628, 3.8032577370081038, -2.725932883755073, 2.404659529699889, 0.9473542902217686, -3.774306842346299, 0.24296746026529314]))</pre>

<pre class='language-julia'><code class='language-julia'>model2 = BDMLModel(Y2, D2, X2, model_type = :hier)</code></pre>
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
    summary(model2);
    coeftable(model2)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash193579">Bayesian Double ML Coefficient Table
======================================================================
Parameter: α (treatment effect)
Model type: hier
Inference method: VI
Credible interval level: 95.0% (HPD)
Number of posterior samples: 2000

  Parameter     Estimate   Std. Error         MCSE      P-value
  ---------     --------   ----------         ----      -------
  α               1.9554       0.0642       0.0000       0.0000

HPD Credible Intervals:
  α: [1.8331, 2.0873]

Diagnostics:
  Final ELBO: -2153.53
</pre>

<!-- PlutoStaticHTML.End -->
```

