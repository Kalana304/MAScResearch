# Multi-node brain model simulation with empirical structural connectivity data

This directory contains the simulations run with structural connectivity data downloaded from https://osf.io/befnx/.

Common parameter setup:

1. Structural connectivity from data. Re-arrange matrix based on weighted degree from highest to the lowest. Normalization: $\max\left\{\sum_{j=1}^N a_{ij}\right\} = 1 \forall i \in [N]$

2. $K_{glob} \in [-1, 1]$

<table style="border-collapse: collapse; border: 1px solid black;">
  <tr>
    <th rowspan="2" align="center">Directory</th>
    <th rowspan="2" align="center">Stimulation</th>
    <th colspan="2" align="center">Heterogeneity</th>
    <th colspan="2" align="center">Initialization</th>
  </tr>
  <tr>
    <th align="center">Stim.</th>
    <th align="center">Non-Stim.</th>
    <th align="center"># Graphs</th>
    <th align="center"># Init Points</th>
  </tr>
  <tr>
    <td>Setup_A1</td>
    <td rowspan="4" align="center">High Deg</td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">10</td>
    <td align="center">1</td>
  </tr>
  <tr>
    <td>Setup_A2</td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">1</td>
    <td align="center">10</td>
  </tr>
  <tr>
    <td>Setup_A3</td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">10</td>
    <td align="center">10</td>
  </tr>
  <tr>
    <td>Setup_A3v</td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">𝒩(σ<sub>m</sub>, σ<sup>2</sup>)</td>
    <td align="center">10</td>
    <td align="center">10</td>
  </tr>
  <tr>
    <td>Setup_B1</td>
    <td rowspan="4" align="center">Low Deg</td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">10</td>
    <td align="center">1</td>
  </tr>
  <tr>
    <td>Setup_B2</td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">1</td>
    <td align="center">10</td>
  </tr>
  <tr>
    <td>Setup_B3</td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">10</td>
    <td align="center">10</td>
  </tr>
  <tr>
    <td>Setup_B3v</td>
    <td align="center">σ<sub>m</sub></td>
    <td align="center">𝒩(σ<sub>m</sub>, σ<sup>2</sup>)</td>
    <td align="center">10</td>
    <td align="center">10</td>
  </tr>
</table>


