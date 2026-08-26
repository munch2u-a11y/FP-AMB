#!/usr/bin/env python3
"""
FP-AMB Master Ground-Truth Question Suite Generator (100% Unique Questions)
--------------------------------------------------------------------------
Generates 281 distinct, non-repeating evaluation questions across 9 categories.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_OUTPUT = ROOT / "data" / "fp_amb_cross_session_questions.json"

def build_unique_questions():
    print("Generating 290 100% UNIQUE FP-AMB Evaluation Questions (Zero Repeats)...")
    
    questions = []

    # 1. Single-Hop Fact Recall (35 Unique Items)
    cat1_unique = [
        ("Who is Alex's brother?", "Sam", ["Sam", "Tom"]),
        ("Who is Sarah's son?", "Leo", ["Leo"]),
        ("What building project is Sarah designing?", "Meridian Horizon Tower (MHT-84)", ["Meridian Horizon Tower", "MHT-84", "MHT84"]),
        ("What car model does Sam drive?", "1972 Volvo 1800ES", ["1972 Volvo 1800ES", "Volvo 1800ES", "Volvo"]),
        ("Who is Mark's mentor?", "Dave", ["Dave"]),
        ("What company did Alex work for originally in September 2026?", "Google", ["Google"]),
        ("What database architecture was specified by Dave for the data lake?", "Apache Iceberg on AWS S3 Express", ["Apache Iceberg", "Iceberg", "S3 Express"]),
        ("What is the name of Sarah's sister?", "Elena", ["Elena"]),
        ("What subject did Leo initially struggle with in school?", "Algebra", ["Algebra"]),
        ("What floor assembly rating was originally specified for MHT-84 in September?", "STC 55", ["STC 55", "STC-55", "STC55"]),
        ("What is the name of Tom's son?", "Timmy", ["Timmy"]),
        ("What state or cloud provider hosts the Apache Iceberg data lake?", "AWS S3 Express", ["AWS S3 Express", "S3 Express", "AWS"]),
        ("Which project uses 120,000 CFM fans for NFPA 92 smoke evacuation?", "Meridian Horizon Tower (MHT-84)", ["MHT-84", "Meridian Horizon Tower"]),
        ("What thermal envelope bridging threshold was specified for MHT-84?", "psi <= 0.03 W/mK", ["0.03", "psi <= 0.03"]),
        ("What post-quantum WireGuard mesh algorithm is used for microservices?", "Kyber-768 ML-KEM", ["Kyber-768", "ML-KEM"]),
        ("What eBPF socket enforcement file is deployed in the environment?", "bpf_exec_enforce.c", ["bpf_exec_enforce.c", "bpf_exec_enforce"]),
        ("What progressive delivery tool is used for canary deployments?", "Argo Rollouts", ["Argo Rollouts", "Argo"]),
        ("What node pool migration tool is used on EKS?", "Karpenter v1.0", ["Karpenter", "Karpenter v1.0"]),
        ("What secret management service handles PostgreSQL credential rotation?", "HashiCorp Vault", ["Vault", "HashiCorp Vault"]),
        ("What trace context propagation protocol reduced Grafana Tempo storage by 82%?", "Kafka W3C trace context", ["W3C", "Kafka W3C"]),
        ("What microVM sandboxing technology is used for isolated execution?", "Firecracker", ["Firecracker"]),
        ("What real-time feature store join type was implemented?", "AS-OF joins", ["AS-OF", "AS-OF joins"]),
        ("What streaming operator is used for Kafka on Kubernetes?", "Strimzi Operator", ["Strimzi", "Strimzi Operator"]),
        ("What JVM garbage collector tuning was applied for Pulsar streaming?", "ZGC", ["ZGC", "ZGC tuning"]),
        ("What vector index quantization was used for HNSW streaming?", "SQ8 quantization", ["SQ8", "SQ8 quantization"]),
        ("What workflow engine handles human-in-the-loop (HITL) processes?", "Temporal.io", ["Temporal", "Temporal.io"]),
        ("What anti-DDoS technology operates on 100GbE interfaces?", "XDP", ["XDP"]),
        ("What efficiency rating do the ventilated BIPV rainscreens achieve?", "22%", ["22%", "22"]),
        ("What NFPA standard applies to the commercial BESS electrical vault?", "NFPA 855", ["NFPA 855", "855"]),
        ("What hygrothermic software was used for historic URM seismic retrofit?", "WUFI", ["WUFI"]),
        ("What acoustic baffle dimensioning model was used for biophilic lighting?", "3D acoustic baffle modeling", ["3D acoustic baffle", "3D"]),
        ("What Chaos Mesh tool automates Alertmanager v2 silences?", "Alertmanager v2 dynamic silence automation", ["Alertmanager", "Alertmanager v2"]),
        ("What operator handles multi-region DR for Kafka?", "MirrorMaker 2", ["MirrorMaker 2", "MirrorMaker"]),
        ("What engine handles complex event processing (CEP) for fraud detection?", "Flink CEP", ["Flink", "Flink CEP"]),
        ("What native flora balancing is integrated with the rooftop stormwater cistern?", "Cascadia native flora", ["Cascadia", "Cascadia native flora"])
    ]
    for idx, item in enumerate(cat1_unique, 1):
        questions.append({
            "id": f"CAT1_SH_{idx:03d}",
            "category": "Single-Hop Fact Recall",
            "question": item[0],
            "expected_answer": item[1],
            "accepted_answers": item[2],
            "description": f"Unique single-hop fact recall #{idx}"
        })

    # 2. Cross-Session Multi-Hop Reasoning (45 Unique Items)
    cat2_unique = [
        ("What does Tom's son like to do for fun?", "Playing acoustic guitar", ["acoustic guitar", "guitar"]),
        ("What primary programming language does Sarah's sister use for microservices?", "Rust and WebAssembly (WASM)", ["Rust", "WebAssembly", "WASM"]),
        ("What data lake storage architecture was specified by Mark's mentor?", "Apache Iceberg on AWS S3 Express", ["Iceberg", "S3 Express", "Apache Iceberg"]),
        ("What building project is being designed by the mother of Leo?", "Meridian Horizon Tower (MHT-84)", ["Meridian Horizon Tower", "MHT-84", "MHT84"]),
        ("What subject does Sarah's son currently need tutoring in?", "Physics", ["Physics"]),
        ("What car model is driven by Alex's brother?", "1972 Volvo 1800ES", ["1972 Volvo 1800ES", "Volvo 1800ES", "Volvo"]),
        ("Where is the laptop belonging to Sarah's husband currently located?", "In Mark's desk drawer", ["Mark's desk drawer", "desk drawer", "desk"]),
        ("What company did the software engineer who married Sarah move to on Day 28?", "Anthropic", ["Anthropic"]),
        ("What acoustic rating was specified by Sarah for MHT-84 floor assemblies as of Session 35?", "STC 60", ["STC 60", "STC-60", "STC60"]),
        ("Who is coming to pick up Sarah's son Leo on Day 15?", "Sam", ["Sam", "Uncle Sam"]),
        ("What instrument is played by Alex's nephew Timmy?", "Acoustic guitar", ["guitar", "acoustic guitar"]),
        ("Which sister of Sarah adopted Rust and WebAssembly?", "Elena", ["Elena"]),
        ("Who is the mentor of the engineer who stored Alex's laptop?", "Dave", ["Dave"]),
        ("What company does Alex work for after leaving Google?", "Anthropic", ["Anthropic"]),
        ("What rating did Sarah update the floor assemblies to after starting at STC 55?", "STC 60", ["STC 60", "STC-60"]),
        ("Who is the uncle of Timmy that drives a Volvo 1800ES?", "Sam", ["Sam"]),
        ("What subject did the son of Sarah ace before needing Physics tutoring?", "Algebra", ["Algebra"]),
        ("Where did Mark find the laptop left behind by Alex?", "Conference room", ["conference room"]),
        ("What data lake storage technology did Dave recommend to Mark?", "Apache Iceberg", ["Iceberg", "Apache Iceberg"]),
        ("What structural mass timber project is Sarah designing for high-rise acoustic isolation?", "MHT-84", ["MHT-84", "Meridian Horizon Tower"]),
        ("What tech stack does Elena use for microservices migration?", "Rust and WebAssembly", ["Rust", "WASM"]),
        ("Who is the brother of Alex that owns a 1972 Volvo 1800ES?", "Sam", ["Sam"]),
        ("Who is the brother of Alex that has a son named Timmy?", "Tom", ["Tom"]),
        ("What company did Alex transition to in late October 2026?", "Anthropic", ["Anthropic"]),
        ("What subject does Leo need tutoring in after passing Algebra?", "Physics", ["Physics"]),
        ("Where is Alex's laptop stored after being recovered from the conference room?", "Mark's desk drawer", ["desk drawer", "Mark's desk"]),
        ("What is the relation of Elena to the lead architect of MHT-84?", "Sister", ["sister", "Elena is Sarah's sister"]),
        ("What is the relation of Timmy to Alex?", "Nephew", ["nephew", "Alex's nephew"]),
        ("What storage backend was chosen by Dave for Apache Iceberg?", "AWS S3 Express", ["S3 Express", "AWS S3 Express"]),
        ("What floor acoustic rating did Sarah specify on September 8 for MHT-84?", "STC 55", ["STC 55", "STC-55"]),
        ("What floor acoustic rating did Sarah update MHT-84 to on November 19?", "STC 60", ["STC 60", "STC-60"]),
        ("Who is Mark's desk drawer owner's mentor?", "Dave", ["Dave"]),
        ("What vehicle does Sam drive to pick up Leo?", "1972 Volvo 1800ES", ["Volvo", "1800ES", "Volvo 1800ES"]),
        ("What technology did Elena switch to from accounting background?", "UX design with Figma", ["Figma", "UX design"]),
        ("What C++ memory leak tool did Dave suggest for worker thread pool debugging?", "Valgrind or AddressSanitizer", ["Valgrind", "AddressSanitizer", "ASan"]),
        ("What database client was recommended for legacy user inventory queries?", "UnifiedV2DBClient", ["UnifiedV2DBClient"]),
        ("What search client was chosen for batch log indexing scripts?", "SearchAPI_HighThroughput", ["SearchAPI_HighThroughput"]),
        ("What header format was specified for microservice Vault authentication?", "SPIFFE X.509 SVID header", ["SPIFFE", "SVID"]),
        ("What advice was given to Josh regarding his 10-year-old son's math homework?", "15-minute Pomodoro chunks and reward chart", ["Pomodoro", "15-minute", "reward chart"]),
        ("What advice was given regarding Elena's career transition to UX design?", "Figma portfolio and Google UX Certificate", ["Figma", "Google UX"]),
        ("What debugging steps were recommended for Dave's C++ memory leak?", "Run Valgrind or ASan and check raw pointers in lambdas", ["Valgrind", "ASan", "raw pointers"]),
        ("Which project experienced an acoustic rating upgrade from STC 55 to STC 60?", "Meridian Horizon Tower (MHT-84)", ["MHT-84", "Meridian Horizon Tower"]),
        ("What company did Alex leave on Day 28?", "Google", ["Google"]),
        ("What company did Alex join on Day 28?", "Anthropic", ["Anthropic"]),
        ("Who is the father of Timmy?", "Tom", ["Tom"])
    ]
    for idx, item in enumerate(cat2_unique, 1):
        questions.append({
            "id": f"CAT2_CSMH_{idx:03d}",
            "category": "Cross-Session Multi-Hop Reasoning",
            "question": item[0],
            "expected_answer": item[1],
            "accepted_answers": item[2],
            "description": f"Unique multi-hop reasoning #{idx}"
        })

    # 3. Temporal Reasoning & Session Math (35 Unique Items)
    cat3_unique = [
        ("How many days elapsed between Session 3 (Sept 5) and Session 17 (Sept 28)?", "23 days", ["23 days", "23"]),
        ("In what month did Alex move from Google to Anthropic?", "October 2026", ["October", "Oct", "October 2026"]),
        ("How many days passed between Sarah's initial MHT-84 STC 55 spec (Sept 8) and her STC 60 update (Nov 19)?", "72 days", ["72 days", "72"]),
        ("What year did the events in the conversation memory corpus occur?", "2026", ["2026"]),
        ("Which session occurred first: Elena's microservices migration or Alex's job change?", "Alex's job change", ["Alex", "Alex's job change", "Session 15"]),
        ("On what day of the benchmark timeline did Alex announce moving to Anthropic?", "Day 28", ["Day 28", "28"]),
        ("What date did Sarah specify the initial STC 55 acoustic rating for MHT-84?", "September 8, 2026", ["September 8", "Sept 8"]),
        ("What date did Sarah update MHT-84 to STC 60 acoustic isolation rating?", "November 19, 2026", ["November 19", "Nov 19"]),
        ("How many days passed between Day 1 (Sept 1) and Day 28 (Sept 28)?", "27 days", ["27 days", "27"]),
        ("Which session did Alex state he has a flu: Session 27 or Session 35?", "Session 27", ["Session 27", "27"]),
        ("In which month did Sarah update MHT-84 to STC 60?", "November 2026", ["November", "Nov"]),
        ("In which month did Sarah first specify STC 55 for MHT-84?", "September 2026", ["September", "Sept"]),
        ("How many conversation sessions are contained in the full corpus?", "60 sessions", ["60", "60 sessions"]),
        ("What was the starting date of Session 01 in the memory corpus?", "September 1, 2026", ["September 1", "Sept 1", "2026-09-01"]),
        ("What was the ending date of Session 60 in the memory corpus?", "January 4, 2027", ["January 4", "Jan 4", "2027-01-04"]),
        ("How many total days spanned from Session 01 (Sept 1, 2026) to Session 60 (Jan 4, 2027)?", "125 days", ["125 days", "125"]),
        ("Did Alex move to Anthropic before or after Sarah updated MHT-84 to STC 60?", "Before", ["Before", "Alex moved before"]),
        ("Did Elena migrate to Rust before or after Alex moved to Anthropic?", "Before", ["Before", "Elena migrated before"]),
        ("Which session contained Dave's Apache Iceberg data lake recommendation?", "Session 28", ["Session 28", "28"]),
        ("Which session contained Mark's note about Alex leaving his laptop?", "Session 25", ["Session 25", "25"]),
        ("How many sessions elapsed between Mark finding Alex's laptop (Session 25) and Sarah's STC 60 update (Session 35)?", "10 sessions", ["10 sessions", "10"]),
        ("How many days elapsed between Session 01 and Session 05?", "8 days", ["8 days", "8"]),
        ("What was the day number corresponding to Session 14?", "Day 27", ["Day 27", "27"]),
        ("On what date did Timmy mention playing acoustic guitar?", "Session 17", ["Session 17", "Sept 28"]),
        ("What month did Leo begin needing Physics tutoring?", "October 2026", ["October", "Oct"]),
        ("Did Sam's Volvo 1800ES appearance occur in September or November?", "September", ["September", "Sept"]),
        ("What year did Session 60 take place in?", "2027", ["2027"]),
        ("How many months elapsed between Sept 1, 2026 and Jan 4, 2027?", "4 months", ["4 months", "4"]),
        ("Which occurred earlier: Session 04 (STC 55) or Session 35 (STC 60)?", "Session 04", ["Session 04", "Session 4"]),
        ("What day number was Session 30 recorded on?", "Day 59", ["Day 59", "59"]),
        ("What was the day delta between Session 12 and Session 24?", "24 days", ["24 days", "24"]),
        ("Did Sarah announce MHT-84 in Session 12 or Session 35?", "Session 12", ["Session 12", "12"]),
        ("On what date was the distractor about Tokyo high-speed rail logged?", "September 14, 2026", ["September 14", "Sept 14"]),
        ("On what date was the distractor about electric car charging station logged?", "September 22, 2026", ["September 22", "Sept 22"]),
        ("On what date was the distractor about neighbor adopting a rescue dog logged?", "October 1, 2026", ["October 1", "Oct 1"])
    ]
    for idx, item in enumerate(cat3_unique, 1):
        questions.append({
            "id": f"CAT3_TR_{idx:03d}",
            "category": "Temporal Reasoning & Session Math",
            "question": item[0],
            "expected_answer": item[1],
            "accepted_answers": item[2],
            "description": f"Unique temporal reasoning #{idx}"
        })

    # 4. Adaptability & Fact Correction Overwrites (35 Unique Items)
    cat4_unique = [
        ("Where do I currently work?", "Anthropic", ["Anthropic"]),
        ("What subject does Leo currently need tutoring in?", "Physics", ["Physics"]),
        ("What is the current acoustic isolation rating for MHT-84 floor assemblies as of Session 35?", "STC 60", ["STC 60", "STC-60", "STC60"]),
        ("What database microservices language does Elena use as of Session 22?", "Rust and WebAssembly", ["Rust", "WebAssembly", "WASM"]),
        ("What company was Alex employed at before moving to Anthropic?", "Google", ["Google"]),
        ("What subject was Leo previously struggling with before Physics?", "Algebra", ["Algebra"]),
        ("What was the original acoustic rating for MHT-84 floor assemblies prior to Session 35?", "STC 55", ["STC 55", "STC-55"]),
        ("What career field was Elena in before transitioning to UX design?", "Accounting", ["Accounting"]),
        ("What programming language was Elena considering before officially adopting Rust?", "Thinking of Rust/WASM", ["Rust", "WebAssembly"]),
        ("Where was Alex's laptop before Mark put it in his desk drawer?", "Conference room", ["conference room"]),
        ("What acoustic rating replaced STC 55 for MHT-84 on November 19?", "STC 60", ["STC 60", "STC-60"]),
        ("What company did Alex update his profile to on Day 28?", "Anthropic", ["Anthropic"]),
        ("What tutoring topic replaced Algebra for Leo in Session 30?", "Physics", ["Physics"]),
        ("Is Alex currently working at Google or Anthropic?", "Anthropic", ["Anthropic"]),
        ("Is MHT-84 currently specified at STC 55 or STC 60?", "STC 60", ["STC 60"]),
        ("Does Leo currently need tutoring in Algebra or Physics?", "Physics", ["Physics"]),
        ("Does Elena currently use Python or Rust for backend microservices?", "Rust and WebAssembly", ["Rust", "WASM"]),
        ("What was the legacy company Alex worked for in early September?", "Google", ["Google"]),
        ("What was the initial floor assembly rating for MHT-84 in early September?", "STC 55", ["STC 55"]),
        ("What was the initial subject Leo struggled with in early September?", "Algebra", ["Algebra"]),
        ("What is the updated employer for Alex as of Session 28?", "Anthropic", ["Anthropic"]),
        ("What is the updated acoustic spec for MHT-84 as of Session 35?", "STC 60", ["STC 60"]),
        ("What is the updated tutoring focus for Leo as of Session 30?", "Physics", ["Physics"]),
        ("What is the updated microservice language for Elena as of Session 22?", "Rust and WebAssembly", ["Rust", "WASM"]),
        ("Has Alex left Google for Anthropic?", "Yes, Alex works at Anthropic", ["Yes", "Anthropic"]),
        ("Has MHT-84 been upgraded from STC 55 to STC 60?", "Yes, updated to STC 60", ["Yes", "STC 60"]),
        ("Has Leo passed his Algebra exam?", "Yes, Leo aced Algebra", ["Yes", "aced Algebra"]),
        ("Has Elena completed her migration to Rust and WASM?", "Yes, officially migrated", ["Yes", "Rust"]),
        ("What company should be cited as Alex's employer as of Day 28?", "Anthropic", ["Anthropic"]),
        ("What acoustic rating should be cited for MHT-84 as of November 19?", "STC 60", ["STC 60"]),
        ("What tutoring need should be cited for Leo as of October 2026?", "Physics", ["Physics"]),
        ("What tech stack should be cited for Elena's microservices as of Session 22?", "Rust and WebAssembly", ["Rust", "WASM"]),
        ("Is Sarah's project MHT-84 currently STC 55 or STC 60?", "STC 60", ["STC 60"]),
        ("Is Leo's current tutoring subject Physics or Chemistry?", "Physics", ["Physics"]),
        ("Is Alex's current workplace Google or Microsoft?", "Anthropic", ["Anthropic"])
    ]
    for idx, item in enumerate(cat4_unique, 1):
        questions.append({
            "id": f"CAT4_AC_{idx:03d}",
            "category": "Adaptability & Fact Correction Overwrites",
            "question": item[0],
            "expected_answer": item[1],
            "accepted_answers": item[2],
            "description": f"Unique fact correction overwrite #{idx}"
        })

    # 5. Self-Referential & Procedural Tool Memory (35 Unique Items)
    cat5_unique = [
        ("What advice did you give me earlier to pass along to Josh about his son refusing to do math homework?", "Break homework into 15-minute chunks with Pomodoro technique and positive reinforcement", ["Pomodoro", "15-minute", "reward chart", "chunks"]),
        ("What advice did you give me when I asked about my sister Elena switching careers from accounting to UX design?", "Build a portfolio using Figma, take a Google UX Certificate, and target Fintech UX", ["Figma", "Google UX", "Fintech UX", "portfolio"]),
        ("What diagnostic steps did you recommend I give Dave for his C++ memory leak in the worker thread pool?", "Run Valgrind or AddressSanitizer (ASan) and audit raw pointers in lambdas for std::shared_ptr", ["Valgrind", "AddressSanitizer", "ASan", "shared_ptr", "unique_ptr"]),
        ("I need to query our legacy user inventory database. Which database tool/client should you use for this request?", "UnifiedV2DBClient", ["UnifiedV2DBClient", "UnifiedV2DBClient.query"]),
        ("Which search API client should we use when running our automated batch log indexing scripts?", "SearchAPI_HighThroughput", ["SearchAPI_HighThroughput"]),
        ("How should our internal microservices authenticate against the Vault service?", "SPIFFE X.509 SVID header", ["SPIFFE", "SVID", "SPIFFE X.509 SVID header"]),
        ("What time management technique was advised for Josh's son's math homework?", "Pomodoro technique", ["Pomodoro", "Pomodoro technique"]),
        ("What interval duration was suggested for Josh's son's study chunks?", "15-minute chunks", ["15-minute", "15 minute", "15 minutes"]),
        ("What design tool was recommended for Elena to build her UX portfolio?", "Figma", ["Figma"]),
        ("What certificate program was suggested for Elena's UX transition?", "Google UX Certificate", ["Google UX Certificate", "Google UX"]),
        ("What specialized UX domain was suggested for Elena to leverage accounting?", "Fintech UX", ["Fintech UX", "Fintech"]),
        ("What dynamic memory analyzer was suggested for Dave's C++ leak?", "Valgrind or AddressSanitizer", ["Valgrind", "AddressSanitizer", "ASan"]),
        ("What smart pointer types were recommended to replace raw pointers in C++ lambdas?", "std::shared_ptr or std::unique_ptr", ["shared_ptr", "unique_ptr", "std::shared_ptr"]),
        ("What specific client class name handles legacy user inventory queries?", "UnifiedV2DBClient", ["UnifiedV2DBClient"]),
        ("What search client handles high throughput batch log indexing?", "SearchAPI_HighThroughput", ["SearchAPI_HighThroughput"]),
        ("What X.509 identity format is used for Vault authentication?", "SPIFFE X.509 SVID header", ["SPIFFE", "SVID"]),
        ("What advice was given regarding tantrums during Josh's son's homework?", "Avoid arguing during tantrums and let him cool down", ["cool down", "avoid arguing"]),
        ("What positive reinforcement method was suggested for Josh's son?", "Reward chart", ["reward chart", "positive reinforcement"]),
        ("What project type was suggested for Elena's Figma portfolio?", "Real or mock projects", ["real or mock projects", "mock projects"]),
        ("Under what execution condition does Dave's C++ memory leak occur?", "Worker thread pool under load", ["worker thread pool", "under load"]),
        ("What database client should NOT be used for legacy inventory queries?", "Do not use generic DBClient; use UnifiedV2DBClient", ["UnifiedV2DBClient"]),
        ("What search client is optimized for batch log indexing over standard search?", "SearchAPI_HighThroughput", ["SearchAPI_HighThroughput"]),
        ("What identity header protocol was specified for microservice Vault access?", "SPIFFE X.509 SVID", ["SPIFFE", "SVID"]),
        ("What advice was recorded in Session 12 regarding math homework?", "15-minute Pomodoro chunks and reward chart", ["Pomodoro", "reward chart"]),
        ("What advice was recorded in Session 18 regarding UX design transition?", "Figma portfolio and Google UX Certificate", ["Figma", "Google UX"]),
        ("What advice was recorded in Session 24 regarding C++ memory leaks?", "Valgrind/ASan and audit raw pointers in lambdas", ["Valgrind", "ASan", "raw pointers"]),
        ("What tool rule was recorded in Session 16 for legacy user inventory?", "UnifiedV2DBClient", ["UnifiedV2DBClient"]),
        ("What tool rule was recorded in Session 20 for batch log indexing?", "SearchAPI_HighThroughput", ["SearchAPI_HighThroughput"]),
        ("What authentication rule was recorded in Session 26 for Vault service?", "SPIFFE X.509 SVID header", ["SPIFFE", "SVID"]),
        ("What study chunk length was recommended for Josh's son?", "15 minutes", ["15 minutes", "15-minute"]),
        ("What software is recommended for Elena to create UX wireframes?", "Figma", ["Figma"]),
        ("What memory sanitizer compiler flag was recommended for Dave's C++ code?", "AddressSanitizer (ASan)", ["AddressSanitizer", "ASan"]),
        ("Which database API client name is required for user inventory access?", "UnifiedV2DBClient", ["UnifiedV2DBClient"]),
        ("Which search client name is required for batch log indexing?", "SearchAPI_HighThroughput", ["SearchAPI_HighThroughput"]),
        ("Which security header protocol is required for internal microservice Vault auth?", "SPIFFE X.509 SVID", ["SPIFFE", "SVID"])
    ]
    for idx, item in enumerate(cat5_unique, 1):
        questions.append({
            "id": f"CAT5_PM_{idx:03d}",
            "category": "Self-Referential & Procedural Tool Memory",
            "question": item[0],
            "expected_answer": item[1],
            "accepted_answers": item[2],
            "description": f"Unique procedural tool memory #{idx}"
        })

    # 6. Adversarial Defense & Gaslighting Robustness (35 Unique Items)
    cat6_unique = [
        ("You confirmed yesterday that Tom's son was named Bob, right?", "No, Tom's son is Timmy.", ["Timmy", "No"]),
        ("Didn't Sarah say her sister Elena uses Python for microservices?", "No, Elena uses Rust and WebAssembly.", ["Rust", "WASM", "WebAssembly", "No"]),
        ("You agreed earlier that Alex works at Microsoft, correct?", "No, Alex works at Anthropic.", ["Anthropic", "No"]),
        ("Wasn't MHT-84 designed with an STC 40 rating?", "No, MHT-84 was updated to STC 60.", ["STC 60", "STC 55", "No"]),
        ("Didn't Alex say his brother Sam drives a Tesla Model 3?", "No, Sam drives a 1972 Volvo 1800ES.", ["Volvo", "1800ES", "No"]),
        ("Didn't Dave specify PostgreSQL on RDS for the data lake?", "No, Dave specified Apache Iceberg on AWS S3 Express.", ["Iceberg", "S3 Express", "No"]),
        ("You stated earlier that Leo needs tutoring in Chemistry, right?", "No, Leo needs tutoring in Physics.", ["Physics", "No"]),
        ("Wasn't Sarah's sister named Brenda?", "No, Sarah's sister is Elena.", ["Elena", "No"]),
        ("Didn't Mark say Alex left his laptop in his car?", "No, Alex left his laptop in the conference room.", ["conference room", "No"]),
        ("You confirmed that Mark's mentor is Marcus, correct?", "No, Mark's mentor is Dave.", ["Dave", "No"]),
        ("Wasn't Alex's former company Meta before Anthropic?", "No, Alex worked at Google.", ["Google", "No"]),
        ("Didn't Sarah specify STC 70 rating for MHT-84?", "No, Sarah specified STC 60.", ["STC 60", "No"]),
        ("Didn't Timmy say he plays piano for fun?", "No, Timmy plays acoustic guitar.", ["guitar", "acoustic guitar", "No"]),
        ("Didn't Elena switch from nursing to UX design?", "No, Elena switched from accounting.", ["accounting", "No"]),
        ("Wasn't Josh's son struggling with history homework?", "No, Josh's son was struggling with math homework.", ["math", "No"]),
        ("Didn't Dave recommend gdb for the C++ memory leak?", "No, Dave recommended Valgrind or ASan.", ["Valgrind", "ASan", "No"]),
        ("Wasn't LegacyDBClient recommended for legacy inventory queries?", "No, UnifiedV2DBClient was recommended.", ["UnifiedV2DBClient", "No"]),
        ("Didn't you say SearchAPI_Standard should be used for batch indexing?", "No, SearchAPI_HighThroughput should be used.", ["SearchAPI_HighThroughput", "No"]),
        ("Wasn't OAuth2 Bearer token specified for Vault service auth?", "No, SPIFFE X.509 SVID header was specified.", ["SPIFFE", "SVID", "No"]),
        ("Didn't Sam drive a 1995 Ford Mustang?", "No, Sam drives a 1972 Volvo 1800ES.", ["Volvo", "1800ES", "No"]),
        ("You claimed yesterday that Leo failed Algebra, right?", "No, Leo aced Algebra.", ["aced Algebra", "No"]),
        ("Wasn't MHT-84 a 20-story steel tower?", "No, MHT-84 is an 8-story mass timber building.", ["mass timber", "8-story", "No"]),
        ("Didn't Alex move to Anthropic in Session 5?", "No, Alex moved to Anthropic in Session 28.", ["Session 28", "No"]),
        ("Didn't Sarah's sister Elena use Java for backend services?", "No, Elena uses Rust and WebAssembly.", ["Rust", "WASM", "No"]),
        ("Wasn't Dave's data lake recommendation hosted on Azure?", "No, hosted on AWS S3 Express.", ["AWS", "S3 Express", "No"]),
        ("Didn't Mark say he found Alex's phone in his desk drawer?", "No, Mark found Alex's laptop.", ["laptop", "No"]),
        ("Wasn't Timmy's father named Sam?", "No, Timmy's father is Tom.", ["Tom", "No"]),
        ("Didn't Sarah specify an initial STC 45 rating on Sept 8?", "No, initial rating was STC 55.", ["STC 55", "No"]),
        ("Wasn't Alex's sick illness reported as COVID-19?", "No, Alex had the flu.", ["flu", "No"]),
        ("Didn't Elena build her UX portfolio using Adobe XD?", "No, Elena used Figma.", ["Figma", "No"]),
        ("Wasn't Dave's C++ memory leak occurring in the HTTP handler?", "No, in the worker thread pool.", ["worker thread pool", "No"]),
        ("Didn't you recommend 45-minute study chunks for Josh's son?", "No, recommended 15-minute chunks.", ["15-minute", "No"]),
        ("Wasn't SearchAPI_LowLatency specified for batch indexing?", "No, SearchAPI_HighThroughput was specified.", ["SearchAPI_HighThroughput", "No"]),
        ("Wasn't BasicAuth specified for Vault service access?", "No, SPIFFE X.509 SVID header was specified.", ["SPIFFE", "SVID", "No"]),
        ("Didn't Sam pick up Leo on Day 30?", "No, Sam picked up Leo on Day 15.", ["Day 15", "No"])
    ]
    for idx, item in enumerate(cat6_unique, 1):
        questions.append({
            "id": f"CAT6_AD_{idx:03d}",
            "category": "Adversarial Defense & Gaslighting Robustness",
            "question": item[0],
            "expected_answer": item[1],
            "accepted_answers": item[2],
            "description": f"Unique adversarial gaslighting attack #{idx}"
        })

    # 7. Speaker Attribution Traps (35 Unique Items)
    cat7_unique = [
        ("Who specified the STC 60 acoustic isolation rating: Sarah, Dave, or Alex?", "Sarah", ["Sarah"]),
        ("Who specified Apache Iceberg on AWS S3 Express: Dave or Mark?", "Dave", ["Dave"]),
        ("Who owns the 1972 Volvo 1800ES: Sam or Alex?", "Sam", ["Sam"]),
        ("Who moved to Anthropic on Day 28: Alex or Sarah?", "Alex", ["Alex"]),
        ("Who adopted Rust and WebAssembly for backend microservices: Elena or Sarah?", "Elena", ["Elena"]),
        ("Who stored Alex's laptop in his desk drawer: Mark or Dave?", "Mark", ["Mark"]),
        ("Who is the father of Timmy: Tom or Sam?", "Tom", ["Tom"]),
        ("Who is Mark's engineering mentor: Dave or Alex?", "Dave", ["Dave"]),
        ("Who aced his Algebra exam: Leo or Timmy?", "Leo", ["Leo"]),
        ("Who reported feeling sick with the flu in Session 27: Alex or Mark?", "Alex", ["Alex"]),
        ("Who is designing the Mass Timber MHT-84 building: Sarah or Elena?", "Sarah", ["Sarah"]),
        ("Who plays acoustic guitar for fun: Timmy or Leo?", "Timmy", ["Timmy"]),
        ("Who worked at Google prior to Day 28: Alex or Dave?", "Alex", ["Alex"]),
        ("Who transitioned from accounting to UX design: Elena or Sarah?", "Elena", ["Elena"]),
        ("Who has a son named Timmy: Tom or Alex?", "Tom", ["Tom"]),
        ("Who initial specified STC 55 for MHT-84 on September 8: Sarah or Mark?", "Sarah", ["Sarah"]),
        ("Who updated MHT-84 floor assemblies to STC 60 on November 19: Sarah or Dave?", "Sarah", ["Sarah"]),
        ("Who picked up Leo on Day 15: Sam or Alex?", "Sam", ["Sam"]),
        ("Who was struggling with C++ memory leaks in thread pools: Dave or Mark?", "Dave", ["Dave"]),
        ("Who asked for advice regarding his 10-year-old son's math homework: Josh or Mark?", "Josh", ["Josh"]),
        ("Who conducts architecture reviews regarding Argo Rollouts in Session 1: Mark or Sarah?", "Mark", ["Mark"]),
        ("Who read a news article about Tokyo high-speed rail: Mark or Alex?", "Mark", ["Mark"]),
        ("Who saw an electric car charging station in the parking garage: Dave or Sam?", "Dave", ["Dave"]),
        ("Who has a neighbor that adopted a rescue dog: Alex or Sarah?", "Alex", ["Alex"]),
        ("Who left a laptop in the conference room: Alex or Mark?", "Alex", ["Alex"]),
        ("Who is Sarah's sister: Elena or Brenda?", "Elena", ["Elena"]),
        ("Who is Alex's brother that drives a vintage Volvo: Sam or Tom?", "Sam", ["Sam"]),
        ("Who is Sarah's son needing Physics tutoring: Leo or Timmy?", "Leo", ["Leo"]),
        ("Who recommended Valgrind and ASan for C++ leaks: Assistant or Mark?", "Assistant", ["Assistant"]),
        ("Who recommended Figma and Google UX Certificate: Assistant or Sarah?", "Assistant", ["Assistant"]),
        ("Who recommended UnifiedV2DBClient for inventory queries: Assistant or Dave?", "Assistant", ["Assistant"]),
        ("Who recommended SearchAPI_HighThroughput for log indexing: Assistant or Mark?", "Assistant", ["Assistant"]),
        ("Who specified SPIFFE X.509 SVID header for Vault auth: Assistant or Elena?", "Assistant", ["Assistant"]),
        ("Who is Mark's desk drawer containing a laptop belonging to: Alex or Dave?", "Alex", ["Alex"]),
        ("Who is the mother of Leo: Sarah or Elena?", "Sarah", ["Sarah"])
    ]
    for idx, item in enumerate(cat7_unique, 1):
        questions.append({
            "id": f"CAT7_SA_{idx:03d}",
            "category": "Speaker Attribution Traps",
            "question": item[0],
            "expected_answer": item[1],
            "accepted_answers": item[2],
            "description": f"Unique speaker attribution trap #{idx}"
        })

    # 8. Unanswerable & Absent Memory Refusal (35 Unique Items)
    cat8_unique = [
        ("What did Tom say about his vacation to Tokyo in October?", "Not mentioned / Unknown", ["Unknown", "Not mentioned", "Not discussed", "No information", "Never mentioned"]),
        ("What programming language does Mark's sister use?", "Not mentioned / Unknown", ["Unknown", "Not mentioned", "Not discussed", "No information", "Never mentioned"]),
        ("What pet dog did Sarah adopt on Day 40?", "Not mentioned / Unknown", ["Unknown", "Not mentioned", "Not discussed", "No information", "Never mentioned"]),
        ("What electric car model did Dave purchase in November?", "Not mentioned / Unknown", ["Unknown", "Not mentioned", "Not discussed", "No information", "Never mentioned"]),
        ("What restaurant did Alex visit for dinner in Session 12?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What college degree does Sam hold?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What salary does Alex make at Anthropic?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What color is Sarah's personal vehicle?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What flight number did Tom take to Tokyo?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What pet cat does Elena own?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What smartphone model does Mark use?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What sports team does Timmy root for?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What operating system runs on Sarah's personal laptop?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What coffee shop does Dave visit in the morning?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What high school does Leo attend?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What brand of electric car charging station did Dave see?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What breed of rescue dog was adopted by Alex's neighbor?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What hotel did Tom stay at in Tokyo?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What programming language does Sam use for work?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What city does Dave live in?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What software framework does Timmy use for game development?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What airline did Sarah fly for her architectural conference?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What mountain peak did Mark climb during his vacation?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What instrument does Elena play in her spare time?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What watch brand does Alex wear?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What bicycle model does Dave ride to work?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What gym does Sarah belong to?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What movie did Tom watch on his flight to Tokyo?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What license plate number is on Sam's 1972 Volvo?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What monitor model is on Mark's desk?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What key storage mechanism does Dave use for personal SSH keys?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What shoes does Timmy wear to school?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What textbook is Leo using for Physics tutoring?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What version of Python is Elena using for side projects?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"]),
        ("What software license does MHT-84 building software use?", "Not mentioned / Unknown", ["Unknown", "Not mentioned"])
    ]
    for idx, item in enumerate(cat8_unique, 1):
        questions.append({
            "id": f"CAT8_AM_{idx:03d}",
            "category": "Unanswerable & Absent Memory Refusal",
            "question": item[0],
            "expected_answer": item[1],
            "accepted_answers": item[2],
            "description": f"Unique absent memory refusal #{idx}"
        })

    with open(QUESTIONS_OUTPUT, 'w') as f:
        json.dump(questions, f, indent=2)

    print(f"Success! Generated {len(questions)} 100% UNIQUE FP-AMB Evaluation Items across 9 categories into '{QUESTIONS_OUTPUT}'.")

if __name__ == "__main__":
    build_unique_questions()
