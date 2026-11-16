# The corpus: papers behind the Quantum Information Agent's tools

113 papers in 11 topics. The list is `papers.json`; `fetch_papers.py` downloads the
arXiv versions and the open-access journal versions; the rest you download through your institute.

**What it is for:** choosing the PDF reader (`pdf_bakeoff.py`) and the embedding model (`eval_retrieval.py`)
on the kind of papers this agent is built for, and the retrieval and answer benchmarks later.

**Licences:** the PDFs stay on your computer (`papers/` is in `.gitignore` and never uploaded to the
Space). Only papers under CC BY, CC BY-SA or CC0 may be redistributed; `papers/REPORT.md` lists them.

The arXiv id in the list is a hint: `fetch_papers.py` checks it against arXiv's record (title and first
author) and searches arXiv when there is no hint or the hint does not match.


## Entanglement, separability, measures and Bell inequalities (22)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| epr1935 | Einstein et al., *Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?* | Phys. Rev. 47, 777 (1935) | - | create_state |
| bell1964 | Bell, *On the Einstein Podolsky Rosen paradox* | Physics Physique Fizika 1, 195 (1964) | - | compute_measures, certify_correlations |
| chsh1969 | Clauser et al., *Proposed Experiment to Test Local Hidden-Variable Theories* | Phys. Rev. Lett. 23, 880 (1969) | - | compute_measures, certify_correlations, plot_fringes |
| cirelson1980 | Cirel'son, *Quantum generalizations of Bell's inequality* | Lett. Math. Phys. 4, 93 (1980) | - | compute_measures |
| werner1989 | Werner, *Quantum states with Einstein-Podolsky-Rosen correlations admitting a hidden-variable model* | Phys. Rev. A 40, 4277 (1989) | - | create_state, mix_states |
| horodecki1995 | Horodecki et al., *Violating Bell inequality by mixed spin-1/2 states: necessary and sufficient condition* | Phys. Lett. A 200, 340 (1995) | - | compute_measures, certify_correlations |
| peres1996 | Peres, *Separability Criterion for Density Matrices* | Phys. Rev. Lett. 77, 1413 (1996) | quant-ph/9604005 | compute_measures, certify_correlations |
| horodecki1996sep | Horodecki et al., *Separability of mixed states: necessary and sufficient conditions* | Phys. Lett. A 223, 1 (1996) | quant-ph/9605038 | compute_measures, certify_correlations |
| horodecki1996info | Horodecki and Horodecki, *Information-theoretic aspects of inseparability of mixed states* | Phys. Rev. A 54, 1838 (1996) | quant-ph/9607007 | compute_measures, plot_bloch_and_correlations |
| bennett1996 | Bennett et al., *Mixed-state entanglement and quantum error correction* | Phys. Rev. A 54, 3824 (1996) | quant-ph/9604024 | compute_measures |
| hill1997 | Hill and Wootters, *Entanglement of a Pair of Quantum Bits* | Phys. Rev. Lett. 78, 5022 (1997) | quant-ph/9703041 | compute_measures |
| wootters1998 | Wootters, *Entanglement of Formation of an Arbitrary State of Two Qubits* | Phys. Rev. Lett. 80, 2245 (1998) | quant-ph/9709029 | compute_measures |
| zyczkowski1998 | Zyczkowski et al., *Volume of the set of separable states* | Phys. Rev. A 58, 883 (1998) | quant-ph/9804024 | compute_measures |
| coffman2000 | Coffman et al., *Distributed entanglement* | Phys. Rev. A 61, 052306 (2000) | quant-ph/9907047 | compute_measures |
| terhal2000 | Terhal, *Bell inequalities and the separability criterion* | Phys. Lett. A 271, 319 (2000) | quant-ph/9911057 | certify_correlations |
| vidal2002 | Vidal and Werner, *Computable measure of entanglement* | Phys. Rev. A 65, 032314 (2002) | quant-ph/0102117 | compute_measures |
| verstraete2002 | Verstraete and Wolf, *Entanglement versus Bell Violations and Their Behavior under Local Filtering Operations* | Phys. Rev. Lett. 89, 170401 (2002) | quant-ph/0112012 | compute_measures |
| plenio2005 | Plenio, *Logarithmic Negativity: A Full Entanglement Monotone That is not Convex* | Phys. Rev. Lett. 95, 090503 (2005) | quant-ph/0505071 | compute_measures |
| plenio2007 | Plenio and Virmani, *An introduction to entanglement measures* | Quantum Inf. Comput. 7, 1 (2007) | quant-ph/0504163 | compute_measures |
| horodecki2009 | Horodecki et al., *Quantum entanglement* | Rev. Mod. Phys. 81, 865 (2009) | quant-ph/0702225 | compute_measures, certify_correlations |
| guhne2009 | Guhne and Toth, *Entanglement detection* | Phys. Rep. 474, 1 (2009) | 0811.2803 | certify_correlations |
| brunner2014 | Brunner et al., *Bell nonlocality* | Rev. Mod. Phys. 86, 419 (2014) | 1303.2849 | compute_measures, certify_correlations |

## Entropies, discord, coherence, fidelity and distances (14)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| uhlmann1976 | Uhlmann, *The "transition probability" in the state space of a *-algebra* | Rep. Math. Phys. 9, 273 (1976) | - | compare_states |
| jozsa1994 | Jozsa, *Fidelity for Mixed Quantum States* | J. Mod. Opt. 41, 2315 (1994) | - | compare_states |
| cerf1997 | Cerf and Adami, *Negative Entropy and Information in Quantum Mechanics* | Phys. Rev. Lett. 79, 5194 (1997) | quant-ph/9512022 | compute_measures |
| fuchs1999 | Fuchs and van de Graaf, *Cryptographic distinguishability measures for quantum-mechanical states* | IEEE Trans. Inf. Theory 45, 1216 (1999) | quant-ph/9712042 | compare_states |
| henderson2001 | Henderson and Vedral, *Classical, quantum and total correlations* | J. Phys. A 34, 6899 (2001) | quant-ph/0105028 | compute_measures |
| ollivier2001 | Ollivier and Zurek, *Quantum Discord: A Measure of the Quantumness of Correlations* | Phys. Rev. Lett. 88, 017901 (2001) | quant-ph/0105072 | compute_measures |
| horodecki2005 | Horodecki et al., *Partial quantum information* | Nature 436, 673 (2005) | quant-ph/0505062 | compute_measures |
| luo2008 | Luo, *Quantum discord for two-qubit systems* | Phys. Rev. A 77, 042303 (2008) | - | compute_measures |
| maziero2009 | Maziero et al., *Classical and quantum correlations under decoherence* | Phys. Rev. A 80, 044102 (2009) | 0905.3396 | compute_measures, apply_local_noise, plot_measures_vs_noise |
| ali2010 | Ali et al., *Quantum discord for two-qubit X states* | Phys. Rev. A 81, 042105 (2010) | 1002.3429 | compute_measures |
| mazzola2010 | Mazzola et al., *Sudden Transition between Classical and Quantum Decoherence* | Phys. Rev. Lett. 104, 200401 (2010) | - | compute_measures, evolve_state |
| modi2012 | Modi et al., *The classical-quantum boundary for correlations: Discord and related measures* | Rev. Mod. Phys. 84, 1655 (2012) | 1112.6238 | compute_measures |
| baumgratz2014 | Baumgratz et al., *Quantifying Coherence* | Phys. Rev. Lett. 113, 140401 (2014) | 1311.0275 | compute_measures |
| streltsov2017 | Streltsov et al., *Colloquium: Quantum coherence as a resource* | Rev. Mod. Phys. 89, 041003 (2017) | 1609.02439 | compute_measures |

## EPR steering (10)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| reid1989 | Reid, *Demonstration of the Einstein-Podolsky-Rosen paradox using nondegenerate parametric amplification* | Phys. Rev. A 40, 913 (1989) | - | certify_correlations |
| wiseman2007 | Wiseman et al., *Steering, Entanglement, Nonlocality, and the Einstein-Podolsky-Rosen Paradox* | Phys. Rev. Lett. 98, 140402 (2007) | quant-ph/0612147 | certify_correlations, plot_sweep |
| jones2007 | Jones et al., *Entanglement, Einstein-Podolsky-Rosen correlations, Bell nonlocality, and steering* | Phys. Rev. A 76, 052116 (2007) | 0709.0390 | certify_correlations |
| cavalcanti2009 | Cavalcanti et al., *Experimental criteria for steering and the Einstein-Podolsky-Rosen paradox* | Phys. Rev. A 80, 032112 (2009) | 0907.1109 | certify_correlations |
| saunders2010 | Saunders et al., *Experimental EPR-steering using Bell-local states* | Nat. Phys. 6, 845 (2010) | 0909.0805 | certify_correlations |
| jevtic2014 | Jevtic et al., *Quantum Steering Ellipsoids* | Phys. Rev. Lett. 113, 020402 (2014) | 1303.4724 | plot_bloch_and_correlations |
| jevtic2015 | Jevtic et al., *Einstein-Podolsky-Rosen steering and the steering ellipsoid* | J. Opt. Soc. Am. B 32, A40 (2015) | 1411.1517 | certify_correlations |
| nguyen2016 | Nguyen and Vu, *Necessary and sufficient condition for steerability of two-qubit states by the geometry of steering outcomes* | Europhys. Lett. 115, 10003 (2016) | 1604.03815 | certify_correlations |
| cavalcanti2017 | Cavalcanti and Skrzypczyk, *Quantum steering: a review with focus on semidefinite programming* | Rep. Prog. Phys. 80, 024001 (2017) | 1604.00501 | certify_correlations |
| uola2020 | Uola et al., *Quantum steering* | Rev. Mod. Phys. 92, 015001 (2020) | 1903.06663 | certify_correlations |

## Noise channels, master equations, decoherence and sudden death (13)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| kraus1971 | Kraus, *General state changes in quantum theory* | Ann. Phys. 64, 311 (1971) | - | get_kraus_operators, apply_kraus_operators |
| choi1975 | Choi, *Completely positive linear maps on complex matrices* | Linear Algebra Appl. 10, 285 (1975) | - | apply_kraus_operators |
| lindblad1976 | Lindblad, *On the generators of quantum dynamical semigroups* | Commun. Math. Phys. 48, 119 (1976) | - | evolve_state |
| gorini1976 | Gorini et al., *Completely positive dynamical semigroups of N-level systems* | J. Math. Phys. 17, 821 (1976) | - | evolve_state |
| lidar1998 | Lidar et al., *Decoherence-Free Subspaces for Quantum Computation* | Phys. Rev. Lett. 81, 2594 (1998) | quant-ph/9807004 | apply_global_noise, evolve_state |
| kwiat2000 | Kwiat et al., *Experimental Verification of Decoherence-Free Subspaces* | Science 290, 498 (2000) | - | apply_global_noise |
| yu2004 | Yu and Eberly, *Finite-Time Disentanglement Via Spontaneous Emission* | Phys. Rev. Lett. 93, 140404 (2004) | quant-ph/0404161 | evolve_state, plot_measures_vs_noise |
| yu2006 | Yu and Eberly, *Quantum Open System Theory: Bipartite Aspects* | Phys. Rev. Lett. 97, 140403 (2006) | quant-ph/0602196 | evolve_state |
| almeida2007 | Almeida et al., *Environment-Induced Sudden Death of Entanglement* | Science 316, 579 (2007) | quant-ph/0701184 | evolve_state, apply_local_noise |
| konrad2008 | Konrad et al., *Evolution equation for quantum entanglement* | Nat. Phys. 4, 99 (2008) | - | apply_local_noise |
| yu2009 | Yu and Eberly, *Sudden Death of Entanglement* | Science 323, 598 (2009) | 0910.1396 | evolve_state |
| aolita2015 | Aolita et al., *Open-system dynamics of entanglement: a key issues review* | Rep. Prog. Phys. 78, 042001 (2015) | 1402.3713 | apply_local_noise, apply_global_noise, evolve_state |
| manzano2020 | Manzano, *A short introduction to the Lindblad master equation* | AIP Adv. 10, 025106 (2020) | 1906.04478 | evolve_state |

## Polarization-entangled photon sources and polarization optics (7)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| jones1941 | Jones, *A New Calculus for the Treatment of Optical Systems. I. Description and Discussion of the Calculus* | J. Opt. Soc. Am. 31, 488 (1941) | - | apply_waveplate, create_single_photon_state |
| kwiat1995 | Kwiat et al., *New High-Intensity Source of Polarization-Entangled Photon Pairs* | Phys. Rev. Lett. 75, 4337 (1995) | - | create_state |
| kwiat1999 | Kwiat et al., *Ultrabright source of polarization-entangled photons* | Phys. Rev. A 60, R773 (1999) | quant-ph/9810003 | create_state, compensate_phase |
| white1999 | White et al., *Nonmaximally Entangled States: Production, Characterization, and Utilization* | Phys. Rev. Lett. 83, 3103 (1999) | quant-ph/9908081 | create_pure_state, reconstruct_state_from_counts |
| kim2006 | Kim et al., *Phase-stable source of polarization-entangled photons using a polarization Sagnac interferometer* | Phys. Rev. A 73, 012316 (2006) | quant-ph/0509219 | create_state |
| fedrizzi2007 | Fedrizzi et al., *A wavelength-tunable fiber-coupled source of narrowband entangled photons* | Opt. Express 15, 15377 (2007) | 0706.2877 | create_state |
| rangarajan2009 | Rangarajan et al., *Optimizing type-I polarization-entangled photons* | Opt. Express 17, 18920 (2009) | - | compensate_phase |

## State tomography, maximum likelihood and error bars (16)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| hradil1997 | Hradil, *Quantum-state estimation* | Phys. Rev. A 55, R1561 (1997) | quant-ph/9609012 | reconstruct_state_from_counts |
| james2001 | James et al., *Measurement of qubits* | Phys. Rev. A 64, 052312 (2001) | quant-ph/0103121 | reconstruct_state_from_counts, reconstruct_single_photon, simulate_tomography |
| rehacek2001 | Rehacek et al., *Iterative algorithm for reconstruction of entangled states* | Phys. Rev. A 63, 040303 (2001) | quant-ph/0009093 | reconstruct_state_from_counts |
| thew2002 | Thew et al., *Qudit quantum-state tomography* | Phys. Rev. A 66, 012303 (2002) | quant-ph/0201052 | reconstruct_state_from_counts |
| lvovsky2004 | Lvovsky, *Iterative maximum-likelihood reconstruction in quantum homodyne tomography* | J. Opt. B 6, S556 (2004) | quant-ph/0311097 | reconstruct_state_from_counts |
| altepeter2005 | Altepeter et al., *Photonic State Tomography* | Adv. At. Mol. Opt. Phys. 52, 105 (2005) | - | reconstruct_state_from_counts, simulate_tomography |
| rehacek2007 | Rehacek et al., *Diluted maximum-likelihood algorithm for quantum tomography* | Phys. Rev. A 75, 042108 (2007) | quant-ph/0611244 | reconstruct_state_from_counts |
| deburgh2008 | de Burgh et al., *Choice of measurement sets in qubit tomography* | Phys. Rev. A 78, 052122 (2008) | 0706.3756 | simulate_tomography |
| gross2010 | Gross et al., *Quantum State Tomography via Compressed Sensing* | Phys. Rev. Lett. 105, 150401 (2010) | 0909.3304 | reconstruct_state_from_counts |
| blumekohout2010a | Blume-Kohout, *Optimal, reliable estimation of quantum states* | New J. Phys. 12, 043034 (2010) | quant-ph/0611080 | reconstruct_state_from_counts, get_tomography_report |
| blumekohout2010b | Blume-Kohout, *Hedged Maximum Likelihood Quantum State Estimation* | Phys. Rev. Lett. 105, 200504 (2010) | 1001.2029 | reconstruct_state_from_counts |
| smolin2012 | Smolin et al., *Efficient Method for Computing the Maximum-Likelihood Quantum State from Measurements with Additive Gaussian Noise* | Phys. Rev. Lett. 108, 070502 (2012) | 1106.5458 | reconstruct_state_from_counts |
| christandl2012 | Christandl and Renner, *Reliable Quantum State Tomography* | Phys. Rev. Lett. 109, 120403 (2012) | 1108.5329 | get_tomography_report |
| schwemmer2015 | Schwemmer et al., *Systematic Errors in Current Quantum State Tomography Tools* | Phys. Rev. Lett. 114, 080403 (2015) | 1310.8465 | get_tomography_report, plot_tomography_fit |
| faist2016 | Faist and Renner, *Practical and Reliable Error Bars in Quantum Tomography* | Phys. Rev. Lett. 117, 010404 (2016) | 1509.06763 | get_tomography_report |
| efron1979 | Efron, *Bootstrap Methods: Another Look at the Jackknife* | Ann. Statist. 7, 1 (1979) | - | get_tomography_report |

## Photonic Bell tests and single-photon detection (6)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| aspect1982 | Aspect et al., *Experimental Realization of Einstein-Podolsky-Rosen-Bohm Gedankenexperiment: A New Violation of Bell's Inequalities* | Phys. Rev. Lett. 49, 91 (1982) | - | plot_fringes, certify_correlations |
| weihs1998 | Weihs et al., *Violation of Bell's Inequality under Strict Einstein Locality Conditions* | Phys. Rev. Lett. 81, 5039 (1998) | quant-ph/9810080 | plot_fringes, certify_correlations |
| giustina2015 | Giustina, *Significant-Loophole-Free Test of Bell's Theorem with Entangled Photons* | Phys. Rev. Lett. 115, 250401 (2015) | 1511.03190 | certify_correlations |
| shalm2015 | Shalm, *Strong Loophole-Free Test of Local Realism* | Phys. Rev. Lett. 115, 250402 (2015) | 1511.03189 | certify_correlations |
| hadfield2009 | Hadfield, *Single-photon detectors for optical quantum information applications* | Nat. Photonics 3, 696 (2009) | - | simulate_tomography |
| eisaman2011 | Eisaman et al., *Invited Review Article: Single-photon sources and detectors* | Rev. Sci. Instrum. 82, 071101 (2011) | - | simulate_tomography |

## Teleportation and the fully entangled fraction (7)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| bennett1993 | Bennett et al., *Teleporting an unknown quantum state via dual classical and Einstein-Podolsky-Rosen channels* | Phys. Rev. Lett. 70, 1895 (1993) | - | teleportation_fidelity |
| popescu1994 | Popescu, *Bell's inequalities versus teleportation: What is nonlocality?* | Phys. Rev. Lett. 72, 797 (1994) | - | teleportation_fidelity |
| massar1995 | Massar and Popescu, *Optimal Extraction of Information from Finite Quantum Ensembles* | Phys. Rev. Lett. 74, 1259 (1995) | - | teleportation_fidelity |
| bouwmeester1997 | Bouwmeester et al., *Experimental quantum teleportation* | Nature 390, 575 (1997) | 1901.11004 | teleportation_fidelity |
| horodecki1999 | Horodecki et al., *General teleportation channel, singlet fraction, and quasidistillation* | Phys. Rev. A 60, 1888 (1999) | quant-ph/9807091 | teleportation_fidelity |
| badziag2000 | Badziag et al., *Local environment can enhance fidelity of quantum teleportation* | Phys. Rev. A 62, 012311 (2000) | quant-ph/9912098 | teleportation_fidelity |
| pirandola2015 | Pirandola et al., *Advances in quantum teleportation* | Nat. Photonics 9, 641 (2015) | 1505.07831 | teleportation_fidelity |

## Quantum key distribution (13)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| bennett1984 | Bennett and Brassard, *Quantum cryptography: Public key distribution and coin tossing* | Theor. Comput. Sci. 560, 7 (reprint of the 1984 paper) (2014) | 2003.06557 | qkd_key_rate |
| ekert1991 | Ekert, *Quantum cryptography based on Bell's theorem* | Phys. Rev. Lett. 67, 661 (1991) | - | qkd_key_rate, certify_correlations |
| bennett1992 | Bennett et al., *Quantum cryptography without Bell's theorem* | Phys. Rev. Lett. 68, 557 (1992) | - | qkd_key_rate |
| shor2000 | Shor and Preskill, *Simple Proof of Security of the BB84 Quantum Key Distribution Protocol* | Phys. Rev. Lett. 85, 441 (2000) | quant-ph/0003004 | qkd_key_rate |
| jennewein2000 | Jennewein et al., *Quantum Cryptography with Entangled Photons* | Phys. Rev. Lett. 84, 4729 (2000) | quant-ph/9912117 | qkd_key_rate |
| naik2000 | Naik et al., *Entangled State Quantum Cryptography: Eavesdropping on the Ekert Protocol* | Phys. Rev. Lett. 84, 4733 (2000) | quant-ph/9912105 | qkd_key_rate |
| gisin2002 | Gisin et al., *Quantum cryptography* | Rev. Mod. Phys. 74, 145 (2002) | quant-ph/0101098 | qkd_key_rate |
| devetak2005 | Devetak and Winter, *Distillation of secret key and entanglement from quantum states* | Proc. R. Soc. A 461, 207 (2005) | quant-ph/0306078 | qkd_key_rate |
| lo2005 | Lo et al., *Efficient Quantum Key Distribution Scheme and a Proof of Its Unconditional Security* | J. Cryptol. 18, 133 (2005) | quant-ph/0011056 | qkd_key_rate |
| ma2007 | Ma et al., *Quantum key distribution with entangled photon sources* | Phys. Rev. A 76, 012307 (2007) | quant-ph/0703122 | qkd_key_rate |
| koashi2009 | Koashi, *Simple security proof of quantum key distribution based on complementarity* | New J. Phys. 11, 045018 (2009) | - | qkd_key_rate |
| scarani2009 | Scarani et al., *The security of practical quantum key distribution* | Rev. Mod. Phys. 81, 1301 (2009) | 0802.4155 | qkd_key_rate |
| pirandola2020 | Pirandola, *Advances in quantum cryptography* | Adv. Opt. Photonics 12, 1012 (2020) | 1906.01645 | qkd_key_rate |

## Quantum error correction (4)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| shor1995 | Shor, *Scheme for reducing decoherence in quantum computer memory* | Phys. Rev. A 52, R2493 (1995) | - | bit_flip_code |
| steane1996 | Steane, *Error Correcting Codes in Quantum Theory* | Phys. Rev. Lett. 77, 793 (1996) | - | bit_flip_code |
| pittman2005 | Pittman et al., *Demonstration of quantum error correction using linear optics* | Phys. Rev. A 71, 052332 (2005) | quant-ph/0502042 | bit_flip_code |
| devitt2013 | Devitt et al., *Quantum error correction for beginners* | Rep. Prog. Phys. 76, 076001 (2013) | 0905.2794 | bit_flip_code |

## Two-qubit state representations (1)

| key | paper | venue | arXiv hint | QI-agent tools |
|---|---|---|---|---|
| fano1983 | Fano, *Pairs of two-level systems* | Rev. Mod. Phys. 55, 855 (1983) | - | plot_bloch_and_correlations, show_density_matrix |
