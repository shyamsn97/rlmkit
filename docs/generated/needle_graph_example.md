# Generated Needle Graph Example

## Collapsed RLM View

This is what the run looks like if recursive calls collapse to strings:

```text
root
  call_llm("scan first third")  -> not found
  call_llm("scan middle third") -> decoy, no code
  call_llm("scan final third")  -> candidate code 84721
  call_llm("verify candidate")  -> 84721 matches the requested needle
  final answer                  -> 84721
```

## Sequence View

This is the same run as calls and returns:

```mermaid
sequenceDiagram
    participant root as root
    participant root_chunk_0 as root.chunk_0
    participant root_chunk_1 as root.chunk_1
    participant root_chunk_2 as root.chunk_2
    participant root_chunk_2_candidate_a as root.chunk_2.candidate_a
    participant root_chunk_2_candidate_b as root.chunk_2.candidate_b
    participant root_verify as root.verify
    root->>+root_chunk_0: delegate Scan first third for the hidden secret code.
    root->>+root_chunk_1: delegate Scan middle third for the hidden secret code.
    root->>+root_chunk_2: delegate Scan final third for the hidden secret code.
    root_chunk_0-->>-root: not found
    root_chunk_1-->>-root: decoy, no code
    root_chunk_2->>+root_chunk_2_candidate_a: delegate Inspect candidate window A.
    root_chunk_2->>+root_chunk_2_candidate_b: delegate Inspect candidate window B.
    root_chunk_2_candidate_a-->>-root_chunk_2: decoy: the code is not 12345
    root_chunk_2_candidate_b-->>-root_chunk_2: needle: the secret code is 84721
    root_chunk_2-->>-root: candidate code 84721
    root->>+root_verify: delegate Verify candidate code 84721 against the origi...
    root_verify-->>-root: 84721 matches the requested needle
    root-->>root: done 84721
```

## Steppable Graph Snapshots

### 1. Root parks after spawning parallel children

```mermaid
flowchart TD
    n_81a48172cd76["root<br/><i>query</i>"]:::query
    n_81a48172cd76 --> n_54f7b70f0df9
    n_54f7b70f0df9["root<br/><i>action</i>"]:::action
    n_54f7b70f0df9 --> n_493176ab4b8d
    n_493176ab4b8d["root<br/><i>supervising</i>"]:::sup
    n_493176ab4b8d --> n_9fb77f6ecec7
    n_9fb77f6ecec7["root.chunk_0<br/><i>query</i>"]:::query
    n_493176ab4b8d --> n_445877af5823
    n_445877af5823["root.chunk_1<br/><i>query</i>"]:::query
    n_493176ab4b8d --> n_93ac8f4c2c65
    n_93ac8f4c2c65["root.chunk_2<br/><i>query</i>"]:::query
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### 2. First children finish while chunk_2 keeps working

```mermaid
flowchart TD
    n_81a48172cd76["root<br/><i>query</i>"]:::query
    n_81a48172cd76 --> n_54f7b70f0df9
    n_54f7b70f0df9["root<br/><i>action</i>"]:::action
    n_54f7b70f0df9 --> n_493176ab4b8d
    n_493176ab4b8d["root<br/><i>supervising</i>"]:::sup
    n_493176ab4b8d --> n_9fb77f6ecec7
    n_9fb77f6ecec7["root.chunk_0<br/><i>query</i>"]:::query
    n_9fb77f6ecec7 --> n_b1037fbaf953
    n_b1037fbaf953["root.chunk_0<br/><i>action</i>"]:::action
    n_b1037fbaf953 --> n_cc5dfa8dd756
    n_cc5dfa8dd756["root.chunk_0<br/><i>result</i><br/>not found"]:::result
    n_493176ab4b8d --> n_445877af5823
    n_445877af5823["root.chunk_1<br/><i>query</i>"]:::query
    n_445877af5823 --> n_77979f531cb4
    n_77979f531cb4["root.chunk_1<br/><i>action</i>"]:::action
    n_77979f531cb4 --> n_71872d4ea370
    n_71872d4ea370["root.chunk_1<br/><i>result</i><br/>decoy, no code"]:::result
    n_493176ab4b8d --> n_93ac8f4c2c65
    n_93ac8f4c2c65["root.chunk_2<br/><i>query</i>"]:::query
    n_93ac8f4c2c65 --> n_e50005447018
    n_e50005447018["root.chunk_2<br/><i>action</i>"]:::action
    n_e50005447018 --> n_640c12be5877
    n_640c12be5877["root.chunk_2<br/><i>supervising</i>"]:::sup
    n_640c12be5877 --> n_60d0d6e4b555
    n_60d0d6e4b555["root.chunk_2.candidate_a<br/><i>query</i>"]:::query
    n_640c12be5877 --> n_06f7abd7013f
    n_06f7abd7013f["root.chunk_2.candidate_b<br/><i>query</i>"]:::query
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### 3. chunk_2 resumes from candidate readers

```mermaid
flowchart TD
    n_81a48172cd76["root<br/><i>query</i>"]:::query
    n_81a48172cd76 --> n_54f7b70f0df9
    n_54f7b70f0df9["root<br/><i>action</i>"]:::action
    n_54f7b70f0df9 --> n_493176ab4b8d
    n_493176ab4b8d["root<br/><i>supervising</i>"]:::sup
    n_493176ab4b8d --> n_9fb77f6ecec7
    n_9fb77f6ecec7["root.chunk_0<br/><i>query</i>"]:::query
    n_9fb77f6ecec7 --> n_b1037fbaf953
    n_b1037fbaf953["root.chunk_0<br/><i>action</i>"]:::action
    n_b1037fbaf953 --> n_cc5dfa8dd756
    n_cc5dfa8dd756["root.chunk_0<br/><i>result</i><br/>not found"]:::result
    n_493176ab4b8d --> n_445877af5823
    n_445877af5823["root.chunk_1<br/><i>query</i>"]:::query
    n_445877af5823 --> n_77979f531cb4
    n_77979f531cb4["root.chunk_1<br/><i>action</i>"]:::action
    n_77979f531cb4 --> n_71872d4ea370
    n_71872d4ea370["root.chunk_1<br/><i>result</i><br/>decoy, no code"]:::result
    n_493176ab4b8d --> n_93ac8f4c2c65
    n_93ac8f4c2c65["root.chunk_2<br/><i>query</i>"]:::query
    n_93ac8f4c2c65 --> n_e50005447018
    n_e50005447018["root.chunk_2<br/><i>action</i>"]:::action
    n_e50005447018 --> n_640c12be5877
    n_640c12be5877["root.chunk_2<br/><i>supervising</i>"]:::sup
    n_640c12be5877 --> n_60d0d6e4b555
    n_60d0d6e4b555["root.chunk_2.candidate_a<br/><i>query</i>"]:::query
    n_60d0d6e4b555 --> n_11c9db178724
    n_11c9db178724["root.chunk_2.candidate_a<br/><i>action</i>"]:::action
    n_11c9db178724 --> n_5a1a96e20084
    n_5a1a96e20084["root.chunk_2.candidate_a<br/><i>result</i><br/>decoy: the code is not 12345"]:::result
    n_640c12be5877 --> n_06f7abd7013f
    n_06f7abd7013f["root.chunk_2.candidate_b<br/><i>query</i>"]:::query
    n_06f7abd7013f --> n_7ffdb2ad836e
    n_7ffdb2ad836e["root.chunk_2.candidate_b<br/><i>action</i>"]:::action
    n_7ffdb2ad836e --> n_d2eab54c28b0
    n_d2eab54c28b0["root.chunk_2.candidate_b<br/><i>result</i><br/>needle: the secret code is 84721"]:::result
    n_640c12be5877 --> n_a6cdbf27cd8e
    n_a6cdbf27cd8e["root.chunk_2<br/><i>result</i><br/>candidate code 84721"]:::result
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### 4. Root resumes and returns the answer

```mermaid
flowchart TD
    n_81a48172cd76["root<br/><i>query</i>"]:::query
    n_81a48172cd76 --> n_54f7b70f0df9
    n_54f7b70f0df9["root<br/><i>action</i>"]:::action
    n_54f7b70f0df9 --> n_493176ab4b8d
    n_493176ab4b8d["root<br/><i>supervising</i>"]:::sup
    n_493176ab4b8d --> n_9fb77f6ecec7
    n_9fb77f6ecec7["root.chunk_0<br/><i>query</i>"]:::query
    n_9fb77f6ecec7 --> n_b1037fbaf953
    n_b1037fbaf953["root.chunk_0<br/><i>action</i>"]:::action
    n_b1037fbaf953 --> n_cc5dfa8dd756
    n_cc5dfa8dd756["root.chunk_0<br/><i>result</i><br/>not found"]:::result
    n_493176ab4b8d --> n_445877af5823
    n_445877af5823["root.chunk_1<br/><i>query</i>"]:::query
    n_445877af5823 --> n_77979f531cb4
    n_77979f531cb4["root.chunk_1<br/><i>action</i>"]:::action
    n_77979f531cb4 --> n_71872d4ea370
    n_71872d4ea370["root.chunk_1<br/><i>result</i><br/>decoy, no code"]:::result
    n_493176ab4b8d --> n_93ac8f4c2c65
    n_93ac8f4c2c65["root.chunk_2<br/><i>query</i>"]:::query
    n_93ac8f4c2c65 --> n_e50005447018
    n_e50005447018["root.chunk_2<br/><i>action</i>"]:::action
    n_e50005447018 --> n_640c12be5877
    n_640c12be5877["root.chunk_2<br/><i>supervising</i>"]:::sup
    n_640c12be5877 --> n_60d0d6e4b555
    n_60d0d6e4b555["root.chunk_2.candidate_a<br/><i>query</i>"]:::query
    n_60d0d6e4b555 --> n_11c9db178724
    n_11c9db178724["root.chunk_2.candidate_a<br/><i>action</i>"]:::action
    n_11c9db178724 --> n_5a1a96e20084
    n_5a1a96e20084["root.chunk_2.candidate_a<br/><i>result</i><br/>decoy: the code is not 12345"]:::result
    n_640c12be5877 --> n_06f7abd7013f
    n_06f7abd7013f["root.chunk_2.candidate_b<br/><i>query</i>"]:::query
    n_06f7abd7013f --> n_7ffdb2ad836e
    n_7ffdb2ad836e["root.chunk_2.candidate_b<br/><i>action</i>"]:::action
    n_7ffdb2ad836e --> n_d2eab54c28b0
    n_d2eab54c28b0["root.chunk_2.candidate_b<br/><i>result</i><br/>needle: the secret code is 84721"]:::result
    n_640c12be5877 --> n_a6cdbf27cd8e
    n_a6cdbf27cd8e["root.chunk_2<br/><i>result</i><br/>candidate code 84721"]:::result
    n_493176ab4b8d --> n_215cf6d35145
    n_215cf6d35145["root<br/><i>resume</i>"]:::resume
    n_215cf6d35145 --> n_e99d458a7106
    n_e99d458a7106["root<br/><i>action</i>"]:::action
    n_e99d458a7106 --> n_031fb854df25
    n_031fb854df25["root<br/><i>supervising</i>"]:::sup
    n_031fb854df25 --> n_f2f05dc175fd
    n_f2f05dc175fd["root.verify<br/><i>query</i>"]:::query
    n_f2f05dc175fd --> n_ae1c8c197c9b
    n_ae1c8c197c9b["root.verify<br/><i>action</i>"]:::action
    n_ae1c8c197c9b --> n_f2f235bce962
    n_f2f235bce962["root.verify<br/><i>result</i><br/>84721 matches the requested needle"]:::result
    n_031fb854df25 --> n_59c98749891d
    n_59c98749891d["root<br/><i>result</i><br/>84721"]:::result
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```
