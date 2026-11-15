# Veilpipe

> Simulating threshold-encrypted mempools inside Layer 2 block times

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-in%20development-orange)
![Course](https://img.shields.io/badge/course-Cryptography-lightgrey)

## Contents

- [Overview](#overview)
- [The problem](#the-problem)
- [What the simulation does](#what-the-simulation-does)
- [How it works](#how-it-works)
- [Designs compared](#designs-compared)
- [Contributions](#contributions)
- [Attacks modelled](#attacks-modelled)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Experiments](#experiments)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Results](#results)
- [References](#references)
- [Team](#team)

## Overview

Veilpipe is a Python simulator built for our Cryptography course project at BITS Pilani Dubai Campus. It models a Layer 2 blockchain that uses an **encrypted mempool** and asks one practical question: **can a decryption committee finish its work inside an L2 block time, and what does it cost in safety and delay when we speed it up?**

It is a simulation. It does not run real cryptography. Decryption times are taken from the published papers listed below.

## The problem

- On most blockchains, pending transactions sit in a public mempool. Bots and block builders can read them and reorder or insert their own transactions (front-running, back-running, sandwich attacks). The profit they make is called **MEV** (Maximal Extractable Value).
- An **encrypted mempool** hides transactions until their position is fixed. A committee of `n` members shares the decryption key, and any `t` of them can decrypt a batch together.
- The catch is time. Decryption takes computation and network round trips, while Layer 2 blocks are short. In our experiments we test block times from 250 ms to 2 s.
- None of the papers in our survey compare their decryption times against L2 block intervals, and a naive speed-up (pipelining) creates a new MEV risk.

**Our questions:**
1. For a given block time, which committee size, threshold and batch size still fit?
2. How much does pipelining help, and what does it cost in user delay and attack exposure?
3. Can we order transactions fairly without extra rounds or trusted hardware?

## What the simulation does

Veilpipe models a pretend L2 and runs it for a few minutes of simulated time, many times, with different settings.

- **Users** send transactions at random times (Poisson arrivals).
- **The sequencer** collects them and seals a batch every block interval.
- **The committee** (N members, split into S subsets) decrypts each batch. Each member's share time grows with the batch size, shares travel with a random network delay, and the batch is revealed once `t` shares arrive and are combined. Some members can be slow or withhold their share.
- **A release rule** decides when a batch may be revealed: immediately, or only after one or two later batches are sealed.
- **Ordering:** after reveal, the batch is shuffled using a seed derived from the decryption output.
- **A toy exchange** (constant product, `x * y = k`) executes the ordered batch so profits and losses can be measured.
- **An attacker** (the sequencer) tries to back-run revealed batches and to wrap whole batches blindly.
- **Metrics:** throughput, end-to-end delay per user (mean, median, 95th and 99th percentile), whether the system stays stable, attack success and profit, and decryption failure rate.

## How it works

```mermaid
flowchart LR
    U[Users send encrypted transactions] --> S[Sequencer seals a batch]
    S --> C[Committee computes and sends shares]
    C --> R{Release rule}
    R --> V[Batch revealed]
    V --> O[Seed from decryption and shuffle]
    O --> E[Execute on toy exchange]
    A[Attacker] -. watches revealed batches .-> V
```

<!-- Add your architecture figure here once saved in docs/ -->
<!-- ![Architecture](docs/architecture.png) -->

## Designs compared

| Design | Idea | Weakness |
|---|---|---|
| **Sequential** | Collect a batch, decrypt it, then start the next one | Slow: one block per collection plus decryption time |
| **Naive pipelining** | Decrypt batch N while batch N+1 is collected, reveal immediately | Fast, but the sequencer sees N while later batches are still open |
| **Veilpipe** | Pipelining plus a release rule, a shuffle seeded by the decryption itself, and committee subsets for load and liveness | Extra delay for users, and back-running is reduced but not removed |

## Contributions

1. **Permutation seed from the threshold decryption.** The shuffle seed is a hash of the combined decryption output and the batch ID. This needs no extra beacon round and no trusted hardware, and we compare it with both options.
2. **Safe operating region.** A chart showing, for each block time, which combinations of committee size, threshold and batch size still decrypt in time.

## Attacks modelled

| Attack | What the attacker does | What it tests |
|---|---|---|
| **A: Back-run** | Reads a revealed batch and puts a trade into the next batch that is still open, to profit from the price move | The release rule and the shuffle |
| **B: Blind wrap** | Puts one trade first and one last in every batch it builds, guessing the direction | The shuffle, and metadata leaks |

## Project structure

```
Veilpipe/
├── README.md
├── INTERFACES.md          function contracts between modules
├── requirements.txt
├── pytest.ini
├── veilpipe/
│   ├── config.py          all settings in one place
│   ├── timing.py          decryption times from the papers, network delay
│   ├── events.py          event queue and clock
│