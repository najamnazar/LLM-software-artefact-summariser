package sumslicellm.messages;

/*
 * Derived from the released SumSlice source of P. W. McBurney and
 * C. McMillan (2013), bundled in this repository at
 *   ORIGINAL_SUMSLICE/sumslice/sumslice/messages/QuickSummaryMessage.java.
 *
 * Unchanged from the released source apart from the package declaration.
 */

/**
 * @author Collin McMillan
 * @since 2013-03-12
 */
public class QuickSummaryMessage extends Message
{
	private String object;
	private String verb;

        public String getObject()
        {
                return object;
        }

        public void setObject(String object)
        {
                this.object = object;
        }

	public String getVerb()
	{
		return verb;
	}

	public void setVerb(String verb)
	{
		this.verb = verb;
	}
}

