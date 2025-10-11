(define (stream guard_rail_streams)
    (:stream next
        :inputs (?p)
        :domain (gen ?p)
        :outputs (?n)
        :certified (and (next ?p ?n) (gen ?n)))

    (:stream harmful
        :inputs (?g)
        :domain (gen ?g)
        :outputs ()
        :certified (harmful ?g))

    (:stream ended
        :inputs (?g)
        :domain (gen ?g)
        :outputs ()
        :certified (ended ?g))
    )