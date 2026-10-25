# Claim selection

The first priority from stress testing is reducing irrelevant claims before they
consume research calls. Selection happens in the existing extraction call; it
does not add another agent or API request.

Each selected quote now needs a `research_basis`: `health_safety`,
`science_environment`, `money_law`, `public_affairs`, `product_capability`, or
`other_material_fact`. The last category has the same materiality requirements.
Python drops candidates marked `skip`, missing this field, or using an unknown
value before saving or dispatching them. The basis is an eligibility decision,
not evidence of truth.

The extractor is instructed to require all three:

- A complete, specific factual assertion.
- Independent public evidence that could realistically settle it.
- A material effect on the viewer's understanding or decisions if false.

It should omit the video's scoreboard, inventory, stunt predictions, private
anecdotes, banter, rhetorical questions, incomplete accusations, incidental
trivia, and overlapping selections of the same assertion within a chunk.
Relevant health or product claims inside entertainment remain eligible.
There is no minimum output quota or hard claim-count cap.

Existing saved `claims.json` files remain usable by standalone verification;
this change does not retroactively reclassify them. Re-extraction uses the new
contract. The mock regression checks that rejected and unclassified candidates
never reach the research callback, while consequential health and public-affairs
claims do. All 28 tests pass.

These checks verify dispatch behavior, not the accuracy of the extractor's
classification. No live paid replay was run. Evaluate the next fresh runs against
a manually labeled set of claims across multiple video genres before declaring
the relevance problem solved. If false positives persist, inspect their selected
bases and refine the gate; do not add research workers to process them faster.
