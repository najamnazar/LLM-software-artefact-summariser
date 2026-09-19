package sumslicellm.messages;

/*
 * Derived from the released SumSlice source of P. W. McBurney and
 * C. McMillan (2013), bundled in this repository at
 *   ORIGINAL_SUMSLICE/sumslice/sumslice/messages/OutputUsedMessage.java.
 *
 * Unchanged from the released source apart from the package declaration.
 */

/**
 * @author Collin McMillan
 * @since 2013-03-12
 */
public class OutputUsedMessage extends Message
{
	private String VP;
	private String NP;

        public String getVP()
        {
                return VP;
        }

        public void setVP(String VP)
        {
                this.VP = VP;
        }

	public String getNP()
	{
		return NP;
	}

	public void setNP(String NP)
	{
		this.NP = NP;
	}
}

