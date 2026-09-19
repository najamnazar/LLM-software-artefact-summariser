package sumslicellm.messages;

/*
 * Derived from the released SumSlice source of P. W. McBurney and
 * C. McMillan (2013), bundled in this repository at
 *   ORIGINAL_SUMSLICE/sumslice/sumslice/messages/ImportanceMessage.java.
 *
 * Unchanged from the released source apart from the package declaration.
 */

/**
 * @author Collin McMillan
 * @since 2013-03-13
 */
public class ImportanceMessage extends Message
{
	private double pagerank;
	private double pagerank_avg;

	public double getPagerank()
	{
		return pagerank;
	}

	public void setPagerank(double pagerank)
	{
		this.pagerank = pagerank;
	}

	public double getAvgPagerank()
	{
		return pagerank_avg;
	}

	public void setAvgPagerank(double avg_pagerank)
	{
		this.pagerank_avg = avg_pagerank;
	}
}

