class RallyState:
    """
    State machine for tracking whether a rally is currently in progress.
    States:
      ACTIVE  — ball is being tracked
      COOLING — ball lost, wait before cutting
      DEAD    — rally over, suppress trail
    """
    def __init__(self, cool_frames=8, revive_frames=2):
        self.cool = cool_frames
        self.revive = revive_frames
        self.state = "DEAD"
        self.miss_ct = 0
        self.hit_ct = 0

    def update(self, detected: bool) -> bool:
        """
        Updates the rally state based on detection.
        Returns True if the trail should be drawn (ACTIVE or COOLING).
        """
        if detected:
            self.miss_ct = 0
            self.hit_ct += 1
            if self.hit_ct >= self.revive:
                self.state = "ACTIVE"
        else:
            self.hit_ct = 0
            self.miss_ct += 1
            if self.miss_ct >= self.cool:
                self.state = "DEAD"
            elif self.state == "ACTIVE":
                self.state = "COOLING"
        
        return self.state in ("ACTIVE", "COOLING")

    @property
    def is_dead(self) -> bool:
        return self.state == "DEAD"

    def reset(self):
        self.state = "DEAD"
        self.miss_ct = 0
        self.hit_ct = 0
