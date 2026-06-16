# The impact of intrinsic excitability heterogeneity on the macro-scale brain dynamics: Computational analysis on stability, resilience, and synchrony

The human brain is a complex network of neurons with diverse properties. Recent computational studies have highlighted the importance of excitability and synaptic heterogeneity on resilience, learning, and memory. However, these studies are confined to cellular-level networks and are inaccessible via non-invasive observation techniques. Local field potentials, commonly used in clinical settings, capture coarse-scale neuronal activities, highlighting the need to understand how neuronal diversity is reflected at a broader spatial scale, or macro-scale. This thesis examines how excitability heterogeneity in neuronal populations affects stability, resilience, and synchronization of macro-scale neural dynamics under external perturbations. We propose a computational
model where each node consists of a neural mass model with interacting excitatory and inhibitory sub-populations. Heterogeneities are represented using lump parameters for each sub-population in each node, and brain region dynamics are coupled through a global synaptic coupling parameter. Our results show that excitability heterogeneity, particularly inhibitory heterogeneity, is critical in maintaining stable and asynchronous neural dynamics, improving resilience against external perturbations, and reducing the likelihood of bifurcations. Variations in the structural connectivity of perturbed regions and changes in heterogeneities in unmodulated nodes influence these results, providing valuable insights for optimizing patient treatment strategies.

## Dynamics of macro-scale node $$n \in [N]$$

<img width="2555" height="1279" alt="Modelsetup" src="https://github.com/user-attachments/assets/421722ed-00e6-46da-b82e-b79758aee832" />

$$
\begin{align*}
  &\tau_e \dot{u}^n_e = -u^n_e + w_{ee}F_{\beta}(u^n_e, \sigma^n_e) + w_{ie}F_{\beta}(u^n_i, \sigma^n_i) + I_{e, 0} + I^n_e + \boxed{{\rm K_{glob}}\sum_{m=1}^N p_{nm}u_e^m}\\
  &\tau_i \dot{u}^n_i = -u^n_i + w_{ei}F_{\beta}(u^n_e, \sigma^n_e) + w_{ii}F_{\beta}(u^n_i, \sigma^n_i) + I_{i, 0} 
\end{align*}
$$





