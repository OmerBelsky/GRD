(define (domain guard_rail_split)
  (:requirements :strips :negative-preconditions :action-costs)

  (:predicates
    (gen ?g)
    (llm-at0 ?g)             (llm-at1 ?g)
    (next0 ?p ?n)            (next1 ?p ?n)
    (next01 ?p ?n)
    (harmful0 ?g)
    (ended1 ?g)
    (found-no-harm1)
    (found-harmful0)
    (split) (done0)
  )

  ;; Cost accumulator
  (:functions (total-cost))

  (:action DoSplit
    :parameters ()
    :precondition (and)
    :effect (and
      (split)
      (increase (total-cost) 0))
  )

  (:action Done0
    :parameters ()
    :precondition (and (split) (found-harmful0))
    :effect (and
      (done0)
      (increase (total-cost) 0))
  )

  ;; -------- Original action: generate(?p ?n) --------
  ;; a0,1: applies before split to BOTH copies
  (:action generate-01
    :parameters (?p ?n)
    :precondition (and
      (not (split))
      (llm-at0 ?p) (llm-at1 ?p)
      (next01 ?p ?n))
    :effect (and
      (llm-at0 ?n) (not (llm-at0 ?p))
      (llm-at1 ?n) (not (llm-at1 ?p))
      (increase (total-cost) 98))
  )

  ;; a0: after split, before Done0 (i.e., not done0)
  (:action generate-0
    :parameters (?p ?n)
    :precondition (and
      (split) (not (done0))
      (llm-at0 ?p)
      (next0 ?p ?n))
    :effect (and
      (llm-at0 ?n) (not (llm-at0 ?p))
      (increase (total-cost) 100))
  )

  ;; a1: after split and Done0
  (:action generate-1
    :parameters (?p ?n)
    :precondition (and
      (split) (done0)
      (llm-at1 ?p)
      (next1 ?p ?n))
    :effect (and
      (llm-at1 ?n) (not (llm-at1 ?p))
      (increase (total-cost) 100))
  )

  (:action mark-found-harm-0
    :parameters (?g)
    :precondition (and
      (split) (not (done0))
      (llm-at0 ?g)
      (harmful0 ?g))
    :effect (and
      (found-harmful0)
      (increase (total-cost) 0))
  )

  (:action mark-found-no-harm-1
    :parameters (?g)
    :precondition (and
      (split) (done0)
      (llm-at1 ?g)
      (ended1 ?g))
    :effect (and
      (found-no-harm1)
      (increase (total-cost) 0))
  )
)
