# Kafka topics (FROZEN, Day 1)

Source: `Claude_Code_Execution_Spec.docx` §3.3. Single broker, both topics on it.
If Kafka needs to shrink under time pressure, collapse to one topic before ever
dropping Kafka itself (see the spec's descope order, §10).

| Topic                   | Producer            | Consumer          | Payload                                       |
|--------------------------|----------------------|--------------------|------------------------------------------------|
| `recommendation-events` | FastAPI `/recommend` | MLflow logger      | `{user_id, track_id, policy, timestamp}`       |
| `user-feedback`         | FastAPI `/feedback`  | Bandit update worker | `{user_id, track_id, action, reward, timestamp}` |

`action` values and their reward mapping (defined in Track C, relevant to any track
consuming `user-feedback`): `completed: +1`, `liked: +2`, `playlist: +3`,
`skip: -1`, `replay: +2`.
