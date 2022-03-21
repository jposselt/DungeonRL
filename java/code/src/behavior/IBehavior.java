package behavior;

public interface IBehavior<INPUT, OUTPUT> {
    OUTPUT nextAction(INPUT state, boolean[] actionMask);
}
