package inel;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ActivationsTest {
    @Test
    void sigmoidRange() {
        assertEquals(0.5, Activations.sigmoid(0.0), 1e-9);
        assertTrue(Activations.sigmoid(-1000) >= 0 && Activations.sigmoid(-1000) <= 1);
        assertTrue(Activations.sigmoid(1000) >= 0 && Activations.sigmoid(1000) <= 1);
    }

    @Test
    void sigmoidNoOverflowForExtremeInputs() {
        assertTrue(Double.isFinite(Activations.sigmoid(-1e6)));
        assertTrue(Double.isFinite(Activations.sigmoid(1e6)));
        assertEquals(0.0, Activations.sigmoid(-1e6), 1e-9);
        assertEquals(1.0, Activations.sigmoid(1e6), 1e-9);
    }

    @Test
    void relu() {
        assertEquals(0.0, Activations.relu(-2.0));
        assertEquals(0.0, Activations.relu(0.0));
        assertEquals(3.5, Activations.relu(3.5));
    }
}
