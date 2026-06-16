# The impact of intrinsic excitability heterogeneity on the macro-scale brain dynamics: Computational analysis on stability, resilience, and synchrony

The human brain is a complex network of neurons with diverse properties. Recent computational studies have highlighted the importance of excitability and synaptic heterogeneity on resilience, learning, and memory. However, these studies are confined to cellular-level networks and are inaccessible via non-invasive observation techniques. Local field potentials, commonly used in clinical settings, capture coarse-scale neuronal activities, highlighting the need to understand how neuronal diversity is reflected at a broader spatial scale, or macro-scale. This thesis examines how excitability heterogeneity in neuronal populations affects stability, resilience, and synchronization of macro-scale neural dynamics under external perturbations. We propose a computational
model where each node consists of a neural mass model with interacting excitatory and inhibitory sub-populations. Heterogeneities are represented using lump parameters for each sub-population in each node, and brain region dynamics are coupled through a global synaptic coupling parameter. Our results show that excitability heterogeneity, particularly inhibitory heterogeneity, is critical in maintaining stable and asynchronous neural dynamics, improving resilience against external perturbations, and reducing the likelihood of bifurcations. Variations in the structural connectivity of perturbed regions and changes in heterogeneities in unmodulated nodes influence these results, providing valuable insights for optimizing patient treatment strategies.

## Dynamics of macro-scale node $$n \in [N]$$

<img width="4116" height="1952" alt="Modelsetup" src="https://github.com/user-attachments/assets/27fe4490-64fb-4227-831e-e5d4701c5642" />

<br> Here we define the mean-field activation function $$F_{\beta}(\cdot, \cdot)$$ as 

$$
  F_{\beta}(x, s) = \underbrace{f_{\beta}(x)}_{\substack{\text{single neuron \\ activation}}} * \underbrace{\varphi(x;s)}_{\text{Gaussian dist.}} = \frac{1}{\sqrt{2\pi s^2}}\int_{-\infty}^{\infty} \frac{1}{1+\exp(-\beta(x + \theta))}e^{-\left(\frac{\theta}{\sqrt{2}s}\right)^2}d\theta.
$$

The following lists the key parameters of our computational model and values used for simulations. 

<table>
<tr>
<td valign="top">

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Excitatory time constant | $\tau_e$ | 10 ms |
| Inhibitory time constant | $\tau_i$ | 5 ms |
| Non-linear gain | $\beta$ | 4.8 |
| Excitatory bias current | $I_e^0$ | -15.625 mV |
| Inhibitory bias voltage | $I_i^0$ | -31.25 mV |
| Stimulation | $I_e^m$ | Variable |
| E-E synaptic strength | $w_{ee}$ | 100 |
| E-I synaptic strength | $w_{ei}$ | 187.5 |
| I-E synaptic strength | $w_{ie}$ | -293.75 |

</td>
<td valign="top">

| Parameter | Symbol | Value |
|-----------|--------|-------|
| I-I synaptic strength | $w_{ii}$ | -8.125 |
| Excitatory heterogeneity | $\sigma_e$ | Variable |
| Inhibitory heterogeneity | $\sigma_i$ | Variable |
| Time step | $\Delta t$ | 0.05 ms |
| Simulation time | $T$ | 2500 ms |
| Settling time | $t_s$ | 500 ms |
| Global coupling | $K$ | Variable |
| Connectivity weights | $p_{nm}$ | Variable |

</td>
</tr>
</table>

### Main modeling assumptions 

- Properties of underlying microcircuits of each macro-scale node are unknown, except the overall diversity of neuronal excitability $$(\sigma_x^n)$$ and the intra-population connectivity $w_{xy}$.
- Only a subset of nodes i.e., $$\mathcal{I}_M \subset [N]$$ is modulated by external modulatory signals: $$I_e^n \neq 0 \Leftrightarrow n \notin \mathcal{I}_M$$
- Only $$\bf{\sigma}^n = [\sigma^n_e, \sigma^n_i]$$ and $$\mathrm{K_{glob}}$$ are changed for each simulation setup. The constant parameters are selected following Scott et al. 2022.
- Time delays between the nodes are ignored (Perl et al. 2023)

### Measuring system behavior

We use the following metrics to quantify the model's behavior under regional varying heterogeneity. 
1. Lyapunov exponents: measure the Lyapunov stability of the regions.
2. Eigenvalues of the Jacobian matrix: measure the system resilience and multi-stable region emergence
3. Functional connectivity using temporal, spectral, and time-frequency measures.

The simulations were carried out first for a simple network with two regions. Later, the analysis was extended to a larger network, with coupling values $p_{nm}$ defined using real-world structural MRI scans. This repository contains the code used to do the numerical simulations:
- Twonode_modeling: Scripts used to simulate the simple two-node network and measure the system behaviour.
- Multinode_empirical: Scripts used to simulate a 10 ROI brain network with coupling values $p_{nm}$ sampled by an appropriate exponential distribution.
- Multinode_statistical: Scripts used to simulate a 10 ROI brain network with coupling values $p_{nm}$ from real-world structural MRI data. 

For details about the simulations, their results, and discussion, please refer to our published work. 

## Citation

If you use this repository in your research, please cite:
```
@article{abeywardena2025impact,
  title={The impact of excitability heterogeneity and synaptic coupling on resilience and stability of a macro-scale brain network},
  author={Abeywardena, Kalana G and Lefebvre, Jeremie and Valiante, Taufik A and Draper, Stark C},
  journal={PLOS Complex Systems},
  volume={2},
  number={7},
  pages={e0000057},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}

@mastersthesis{abeywardena2024impact,
  title={The impact of intrinsic excitability heterogeneity on the macro-scale brain dynamics: Computational analysis on stability, resilience, and synchrony},
  author={Abeywardena, Kalana Gayal},
  year={2024},
  school={University of Toronto (Canada)}
}
```



