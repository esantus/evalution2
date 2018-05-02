package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="source")
public class Source {
	private Long source_id;
	private String source_value;
	
	
	public Source() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public Source(Long source_id, String source_value) {
		super();
		this.source_id = source_id;
		this.source_value = source_value;
	}

	@Id
	@Column(name = "source_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getSource_id() {
		return source_id;
	}

	public void setSource_id(Long source_id) {
		this.source_id = source_id;
	}

	public String getSource_value() {
		return source_value;
	}

	public void setSource_value(String source_value) {
		this.source_value = source_value;
	}

	@Override
	public String toString() {
		return "Source [source_id=" + source_id + ", source_value=" + source_value + "]";
	}
	
	
}
