(define (stream guard_rail_split_streams)

  ;; NEXT generators for each copy
  (:stream next0
    :inputs (?p)
    :domain (gen ?p)
    :outputs (?n)
    :certified (and (next0 ?p ?n) (gen ?n)))

  (:stream next1
    :inputs (?p)
    :domain (gen ?p)
    :outputs (?n)
    :certified (and (next1 ?p ?n) (gen ?n)))

  (:stream next01
    :inputs (?p)
    :domain (gen ?p)
    :outputs (?n)
    :certified (and (next01 ?p ?n) (gen ?n)))

  (:stream harmful0
    :inputs (?g)
    :domain (gen ?g)
    :outputs ()
    :certified (harmful0 ?g))

  (:stream ended1
    :inputs (?g)
    :domain (gen ?g)
    :outputs ()
    :certified (ended1 ?g))
)
